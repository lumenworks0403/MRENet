"""
HED 并行解码器核心逻辑（无依赖 groundingdino）。
从 rrs/model.py 提取，便于单元测试直接导入。

参考: tmp/hed_reference/grounding_dino_layers_HED.py
      GroundingDinoTransformerDecoder_parallel_15_DNQueryRand.forward()
算法来源: FT-FSOD 论文第3.2节, 图2, 公式5-7
本文件实现 HED 并行解码，以及训练时可选的 DN 查询分支。
"""
import torch

from .tensor_ops import inverse_sigmoid


def _compute_ref_points_input(reference_points, valid_ratios):
    """把参考框 (nq,B,4) 扩展为多尺度格式 (nq,B,nlevel,4)。
    与 B0 transformer.py TransformerDecoder.forward 中同名逻辑一致。"""
    if reference_points.shape[-1] == 4:
        return reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
    else:
        return reference_points[:, :, None] * valid_ratios[None, :]


def _run_decoder_layer(layer, output, tgt_query_pos, tgt_query_sine_embed, tgt_key_padding_mask,
                       tgt_reference_points, memory_text, text_attention_mask, memory,
                       memory_key_padding_mask, level_start_index, spatial_shapes, memory_pos,
                       self_attn_mask):
    """调用一个解码器层，参数布局与 B0 transformer.py TransformerDecoder 的 layer() 调用一致。"""
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
    """HED 无DN结构版并行解码器。

    第0层: 串行（共享层），行为与 B0 完全相同，保存输出和参考框。
    第1~5层: 并行层，各自独立从共享状态（层0输出）开始，互不传递中间结果。
      - 输入查询 = shared_output（各层读取相同值，不 clone；各层有独立参数）
      - 输入参考框 = shared_ref（来自层0 detach 后的更新参考框）
      - 各层产生各自的 new_ref（通过各层独立的 bbox_embed）
      - 一个分支的输出不进入另一个分支

    梯度说明:
      - 层0 reference_points 更新后 detach()（与原 B0 一致，防止 box refine 产生二阶梯度）
      - shared_output 不 detach（共享层参数获得来自所有并行层的梯度）
      - 各并行层从 shared_output 读取，梯度经由 shared_output 流回层0参数

    参数:
        decoder: 原 B0 的 TransformerDecoder 对象
                 需要属性: .layers, .norm, .bbox_embed, .ref_point_head
        tgt: (nq, B, C) nq-first 张量（与 B0 transformer 接口一致）
        gen_sineembed_for_position: groundingdino.models.GroundingDINO.utils 的同名函数
                                    (传入以避免此文件直接依赖 groundingdino)

    返回:
        hs:        list[6个(B,nq,C)]，与 B0 hs 格式一致
        refs:      list[7个(B,nq,4)] sigmoid 坐标，第0个为初始参考框，
                   其余为各层自己更新出的参考框（并行层之间互不相关）
        input_refs: list[6个(B,nq,4)] sigmoid 坐标，input_refs[i] 是第 i 层
                   「实际作为输入消费」的参考框。串行结构下它等于 refs[i]，
                   但并行结构下 input_refs[1..5] 全部等于共享层的参考框。
                   注意: 这里的框已 detach, 只用于记录/校验 decoder 实际输入。
                   模型外部的框预测头应使用数值相同但未 detach 的
                   [refs[0]] + [refs[1]]*5（见 model.py::_forward_hed），
                   以保留与 B0 相同的跨层框梯度；也不能用 refs[:-1]
                   （那是串行假设，会让第3层误用第2层分支的位置）。
    """
    reference = refpoints_unsigmoid.sigmoid()
    shared_output, shared_ref = tgt, reference
    intermediate, ref_points, input_refs = [], [reference], []

    for layer_id, layer in enumerate(decoder.layers):
        # 所有并行层都从共享层输出开始，不消费其他分支的结果。
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
            # 返回值保留框预测梯度；传入后续分支的参考框 detach。
            shared_ref = new_ref.detach() if decoder.bbox_embed is not None else new_ref

    return tuple([tensor.transpose(0, 1) for tensor in values]
                 for values in (intermediate, ref_points, input_refs))


# ======================================================================
# HED + DN (训练专用)
# 迁移自官方 GroundingDinoTransformerDecoder_parallel_15_DNQueryRand.forward
# (grounding_dino_layers_HED.py 第 315-496 行) 与
# GroundingDINO_ParallelDecoder_15_DNQuery_rand.pre_decoder
# (grounding_dino_HED.py 第 393-431 行)。
# 上面的 hed_parallel_decode 保持原样不动: 推理、DN 关闭、或整个 batch 无目标时仍走它,
# 因此 HED-structure/no-DN 的行为不受本节影响。
# ======================================================================
BRANCH_REF_MODES = ("sigmoid", "official_raw")


def hed_parallel_decode_dn(decoder, tgt, memory, valid_ratios, spatial_shapes, level_start_index,
                           memory_key_padding_mask, refpoints_unsigmoid, memory_text,
                           text_attention_mask, memory_pos, gen_sineembed_for_position, dn0,
                           branch_dns, branch_ref_mode="sigmoid"):
    """HED + DN 并行解码。

    序列排列 (与官方相同): [DN 查询 (Ndn 个) | 普通查询 (Q 个)], nq-first。

    第0层 (共享层): 输入 = [dn0 | tgt], 参考框 = sigmoid([dn0.bbox_unsig | refpoints]),
        self-attn mask = dn0.attn_mask。普通查询部分与 no-DN 版逐元素相同
        (mask 使普通查询看不到 DN, 其余算子逐查询独立)。
    第1~5层 (并行分支), branch_dns[m] 决定第 m+1 层:
        None    -> 继承: 输入 = 共享层的完整输出 (DN 部分 + 普通部分), mask = dn0.attn_mask
                   (官方 "inherit all queries from the shared layer")。
        DNQueries -> 重新生成: 输入 = [该分支自己的 DN | 共享层输出的普通部分],
                   参考框 = [该分支 DN 的框 | 共享层更新后的普通参考框],
                   mask = 该分支自己的 attn_mask。
        普通查询部分在两种情况下都等于共享层输出, 与 no-DN 版相同。

    branch_ref_mode:
        "sigmoid"      (默认) 分支自己 DN 的参考框 = sigmoid(bbox_unsig), 与共享层 DN
                       (官方在 pre_decoder 里对拼接后的整体做了 sigmoid) 一致。
        "official_raw" 逐字照搬官方: 官方 pre_decoder 把生成器输出的 inverse_sigmoid 值
                       直接存入 dn_bbox_querys, decoder 第 435 行把它与 [0,1] 坐标拼接
                       而未做 sigmoid, 使重新生成的 DN 参考框落在 logit 值域。
                       保留此选项只为对照官方代码的实际行为。

    返回 dict (除 layer_dn 外均为 B-first):
        hs:         list[6×(B,Ndn+Q,C)]
        refs:       list[7×(B,Ndn+Q,4)]  第0个为初始框, 其余为各层自己更新出的框 (未 detach)
        input_refs: list[6×(B,Ndn+Q,4)]  各层 decoder 实际消费的参考框 (已 detach)
        head_refs:  list[6×(B,Ndn+Q,4)]  预测头应使用的参考框: 与 input_refs 数值相同,
                    但普通部分与共享层 DN 部分保留未 detach 的梯度 (同 no-DN 版的修复)。
        layer_dn:   list[6×DNQueries]    第 i 层 DN 部分对应的生成结果 (监督按它定位)
        num_dn:     int                  各层 DN 查询数 (同一 batch GT 相同, 故相同)
    """
    assert branch_ref_mode in BRANCH_REF_MODES, branch_ref_mode
    assert decoder.bbox_embed is not None, "HED+DN 需要 iterative box refine (decoder.bbox_embed)"
    layers = decoder.layers
    assert len(branch_dns) == len(layers) - 1, "每个并行分支需要一个 DN 决定 (None=继承)"
    ndn = dn0.num_queries
    for d in branch_dns:
        if d is not None:
            # 同一 batch 的 GT 不变, 所以重新生成的 DN 数量与布局必须与 dn0 相同;
            # 官方 head 只用 dn_meta_0 切分各层输出, 依赖的正是这一点。
            assert d.num_queries == ndn and d.meta["num_target_list"] == dn0.meta["num_target_list"]

    norm, ref_head, bbox_embed = decoder.norm, decoder.ref_point_head, decoder.bbox_embed

    def _branch_dn_ref(d):
        r = d.bbox_unsig.transpose(0, 1)
        return r.sigmoid() if branch_ref_mode == "sigmoid" else r

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

    # ---- 第0层: 共享层 ----
    q0 = torch.cat([dn0.label_query.transpose(0, 1).to(tgt.dtype), tgt], 0)  # (Ndn+Q, B, C)
    r0 = torch.cat(
        [dn0.bbox_unsig.transpose(0, 1).to(refpoints_unsigmoid.dtype), refpoints_unsigmoid],
        0).sigmoid()  # (Ndn+Q, B, 4)
    shared_out, shared_ref = _run(0, q0, r0, dn0.attn_mask)
    shared_ref_det = shared_ref.detach()  # 与 B0 / no-DN 版相同: 只有喂给下一层的那份 detach

    hs, refs, input_refs, head_refs, layer_dn = [norm(shared_out)], [r0,
                                                                     shared_ref], [r0], [r0], [dn0]

    # ---- 第1~5层: 并行分支 ----
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

    t = lambda xs: [x.transpose(0, 1) for x in xs]  # noqa: E731
    return dict(hs=t(hs), refs=t(refs), input_refs=t(input_refs), head_refs=t(head_refs),
                layer_dn=layer_dn, num_dn=ndn)


def sample_branch_dns(generator, gt_boxes_list, num_branches, rate, rng):
    """官方 pre_decoder 第 405-423 行: 每个并行分支独立地以概率 rate (=论文 τ) 决定
    重新生成自己的 DN (重新调用生成器 -> 新噪声、新 mask、新 meta), 否则继承共享层 DN。

    rate=0: 全部继承 (仍有 DN, 只是不重新初始化); rate=1: 全部重新生成。
    rng: 带 .random() 的对象 (官方用 Python random 模块)。
    与官方的差异: 官方为 6 个槽位都抽签并生成, 只用前 5 个;
    这里只为实际使用的 num_branches 个分支抽签, 不影响算法, 只改变随机数流。
    """
    out = []
    for _ in range(num_branches):
        out.append(generator(gt_boxes_list) if rng.random() < rate else None)
    return out
