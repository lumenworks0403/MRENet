"""SPED 共享并行解码，参照方法第 3.4 节、公式 (6)-(11)。
保留 hed_parallel_decode 函数名以兼容现有入口。"""
import torch

from .tensor_ops import inverse_sigmoid


def _compute_ref_points_input(reference_points, valid_ratios):
    """将归一化参考框扩展到多尺度有效图像区域。"""
    if reference_points.shape[-1] == 4:
        return reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
    else:
        return reference_points[:, :, None] * valid_ratios[None, :]


def _run_decoder_layer(layer, output, tgt_query_pos, tgt_query_sine_embed, tgt_key_padding_mask,
                       tgt_reference_points, memory_text, text_attention_mask, memory,
                       memory_key_padding_mask, level_start_index, spatial_shapes, memory_pos,
                       self_attn_mask):
    """调用一个 GroundingDINO 解码层。"""
    return layer(
        tgt=output,
        tgt_query_pos=tgt_query_pos,
        tgt_query_sine_embed=tgt_query_sine_embed,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_reference_points=tgt_reference_points,
        memory_text=memory_text,
        text_attention_mask=text_attention_mask,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
        memory_level_start_index=level_start_index,
        memory_spatial_shapes=spatial_shapes,
        memory_pos=memory_pos,
        self_attn_mask=self_attn_mask,
        cross_attn_mask=None,
    )


def hed_parallel_decode(decoder, tgt, memory, valid_ratios, spatial_shapes, level_start_index,
                        memory_key_padding_mask, refpoints_unsigmoid, memory_text,
                        text_attention_mask, memory_pos, gen_sineembed_for_position,
                        tgt_key_padding_mask=None, tgt_mask=None):
    """共享层输出 Hs；五个分支独立读取 Hs 和停止梯度后的 Rs。
    返回六组归一化查询、更新框及各层实际输入框。预测头保留共享参考框的梯度。"""
    reference = refpoints_unsigmoid.sigmoid()
    shared_output, shared_ref = tgt, reference
    intermediate, ref_points, input_refs = [], [reference], []

    for layer_id, layer in enumerate(decoder.layers):

        input_refs.append(shared_ref)
        rpi = _compute_ref_points_input(shared_ref, valid_ratios)
        qse = gen_sineembed_for_position(rpi[:, :, 0, :])
        output = _run_decoder_layer(
            layer,
            shared_output,
            decoder.ref_point_head(qse),
            qse,
            tgt_key_padding_mask,
            rpi,
            memory_text,
            text_attention_mask,
            memory,
            memory_key_padding_mask,
            level_start_index,
            spatial_shapes,
            memory_pos,
            tgt_mask if layer_id == 0 else None,
        )
        if decoder.bbox_embed is not None:
            delta = decoder.bbox_embed[layer_id](output)
            new_ref = (delta + inverse_sigmoid(shared_ref)).sigmoid()
        else:
            new_ref = shared_ref if layer_id == 0 else shared_ref.clone()
        intermediate.append(decoder.norm(output))
        ref_points.append(new_ref)
        if layer_id == 0:
            shared_output = output

            shared_ref = new_ref.detach() if decoder.bbox_embed is not None else new_ref

    return tuple([tensor.transpose(0, 1) for tensor in values]
                 for values in (intermediate, ref_points, input_refs))


BRANCH_REF_MODES = ("sigmoid", )


def hed_parallel_decode_dn(decoder, tgt, memory, valid_ratios, spatial_shapes, level_start_index,
                           memory_key_padding_mask, refpoints_unsigmoid, memory_text,
                           text_attention_mask, memory_pos, gen_sineembed_for_position, dn0,
                           branch_dns, branch_ref_mode="sigmoid"):
    """训练序列为 [DN | 普通查询]。分支继承共享 DN，或使用新生成的 DN。
    所有分支的普通查询均读取共享状态；再生参考框必须经过 sigmoid。
    head_refs 保留框预测梯度，input_refs 为送入下一层的停止梯度参考框。"""
    if branch_ref_mode != "sigmoid":
        raise ValueError("再生 DN 参考框必须使用 sigmoid 归一化")
    assert decoder.bbox_embed is not None, "HED+DN 需要 iterative box refine (decoder.bbox_embed)"
    layers = decoder.layers
    assert len(branch_dns) == len(layers) - 1, "每个并行分支需要一个 DN 决定 (None=继承)"
    ndn = dn0.num_queries
    for d in branch_dns:
        if d is not None:

            assert d.num_queries == ndn and d.meta["num_target_list"] == dn0.meta["num_target_list"]

    norm, ref_head, bbox_embed = decoder.norm, decoder.ref_point_head, decoder.bbox_embed

    def _branch_dn_ref(d):
        r = d.bbox_unsig.transpose(0, 1)
        return r.sigmoid()

    def _run(layer_id, q_in, r_in, mask):
        rpi = _compute_ref_points_input(r_in, valid_ratios)
        qse = gen_sineembed_for_position(rpi[:, :, 0, :])
        out = _run_decoder_layer(
            layers[layer_id],
            q_in,
            ref_head(qse),
            qse,
            None,
            rpi,
            memory_text,
            text_attention_mask,
            memory,
            memory_key_padding_mask,
            level_start_index,
            spatial_shapes,
            memory_pos,
            mask,
        )
        new_ref = (bbox_embed[layer_id](out) + inverse_sigmoid(r_in)).sigmoid()
        return out, new_ref

    q0 = torch.cat([dn0.label_query.transpose(0, 1).to(tgt.dtype), tgt], 0)
    r0 = torch.cat(
        [dn0.bbox_unsig.transpose(0, 1).to(refpoints_unsigmoid.dtype), refpoints_unsigmoid],
        0).sigmoid()
    shared_out, shared_ref = _run(0, q0, r0, dn0.attn_mask)
    shared_ref_det = shared_ref.detach()

    hs, refs, input_refs, head_refs, layer_dn = [norm(shared_out)], [r0,
                                                                     shared_ref], [r0], [r0], [dn0]

    for lid in range(1, len(layers)):
        own = branch_dns[lid - 1]
        if own is None:
            q_in, r_in, r_head, mask, used = shared_out, shared_ref_det, shared_ref, dn0.attn_mask, dn0
        else:
            own_ref = _branch_dn_ref(own).to(shared_ref.dtype)
            q_in = torch.cat(
                [own.label_query.transpose(0, 1).to(shared_out.dtype), shared_out[ndn:]], 0)
            r_in = torch.cat([own_ref, shared_ref_det[ndn:]], 0)
            r_head = torch.cat([own_ref, shared_ref[ndn:]], 0)
            mask, used = own.attn_mask, own
        out, new_ref = _run(lid, q_in, r_in, mask)
        hs.append(norm(out))
        refs.append(new_ref)
        input_refs.append(r_in)
        head_refs.append(r_head)
        layer_dn.append(used)

    t = lambda xs: [x.transpose(0, 1) for x in xs]
    return dict(hs=t(hs), refs=t(refs), input_refs=t(input_refs), head_refs=t(head_refs),
                layer_dn=layer_dn, num_dn=ndn)


def sample_branch_dns(generator, gt_boxes_list, num_branches, rate, rng):
    """每个分支为整批 DN 输入独立抽取一次 Bernoulli(rate)；rate=0 仍保留 DN 监督。"""
    out = []
    for _ in range(num_branches):
        out.append(generator(gt_boxes_list) if rng.random() < rate else None)
    return out
