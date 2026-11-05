"""HED+DN 服务器短测试: 真实模型、真实数据, 最多 2 个 batch 前向/反向后自动退出。

不保存任何权重, 不写训练日志, 不调用 optimizer.step (参数不变), 不做全量评估。

    python tools/smoke_dn.py --config configs/b0_hed_dn_rislad.py [--iters 2] [--opts key=value ...]

检查 (结果打印为 JSON, 任一失败则 exit 1):
  1. 训练前向产生 dn_outputs; 各层 DN 数与该层 metadata 一致; 普通查询数 = num_queries。
  2. 总损失与全部 DN 损失有限; 反传后共享层 / 并行分支 / dn_generator / MaskHead 梯度有限,
     且 decoder 第0层、第5层与 label_embedding 梯度非零。
  3. eval 前向不需要 targets, 不含 dn_outputs; 同一份权重在 use_dn=True / False 两个模型上
     (load_model_state 加载同一 state_dict) eval 输出逐元素一致。
  4. 保存/加载往返 (内存中, 不写盘): state_dict 与 cfg 中 use_hed / use_dn / dn 正确恢复。
"""
import argparse
import io
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.criterion import build_criterion  # noqa: E402
from rrs.data import RefSegDataset, collate_fn  # noqa: E402
from rrs.model import build_model, load_model_state  # noqa: E402
from rrs.model_mode import model_variant, resolve_model_mode  # noqa: E402
from rrs.config import load_cfg  # noqa: E402
from rrs.data import load_samples  # noqa: E402
from rrs.training import to_device  # noqa: E402


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*")
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()
    assert 1 <= args.iters <= 2, "短测试最多 2 个 batch"
    cfg = resolve_model_mode(load_cfg(args.config, args.opts), None, args.opts)
    assert cfg["use_hed"] and cfg["use_dn"], "smoke_dn 只用于 HED+DN 配置"
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["amp"] and device == "cuda"
    report, ok = {
        "variant": model_variant(cfg),
        "dn": cfg["dn"],
        "device": device,
        "amp": amp
    }, True

    model, info = build_model(cfg["gdino_cfg"], cfg["gdino_ckpt"], cfg["bert_path"], device,
                              use_checkpoint=cfg["use_checkpoint"], use_hed=True, use_dn=True,
                              dn_cfg=cfg["dn"])
    report["pretrained_missing_keys"] = info.missing_keys if info else None
    criterion = build_criterion(cfg)
    ds = RefSegDataset(load_samples(cfg["train_data"]), cfg["train_data"]["image_root"], train=True,
                       short_side=cfg["short_side"], max_size=cfg["max_size"], hflip=cfg["hflip"])
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                         num_workers=0, collate_fn=collate_fn)
    nq = model.gd.num_queries

    model.train()
    for it, (imgs, targets) in enumerate(loader):
        if it >= args.iters:
            break
        model.zero_grad(set_to_none=True)
        imgs = [i.to(device) for i in imgs]
        targets = to_device(targets, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(imgs, [t["caption"] for t in targets], targets)
        loss, ld, _ = criterion(out, targets)
        loss.backward()
        r = {
            "iter": it,
            "num_targets": [len(t["boxes"]) for t in targets],
            "num_matching": out["pred_logits"].shape[1],
            "loss": loss.item()
        }
        ok &= r["num_matching"] == nq and math.isfinite(r["loss"])
        dn = out.get("dn_outputs")
        if dn is None:
            r["dn"] = "无 (batch 内无目标)"
        else:
            ndn = dn["pred_logits"].shape[2]
            r["num_dn"] = ndn
            r["layer_dn_queries"] = [m["num_denoising_queries"] for m in dn["layer_meta"]]
            ok &= all(n == ndn for n in r["layer_dn_queries"])
            r["dn_losses_last"] = {
                k: round(v.item(), 4)
                for k, v in ld.items() if k.endswith("_dn")
            }
            ok &= all(math.isfinite(v.item()) for k, v in ld.items() if "_dn" in k)
        grads = {}
        for name, mod in (("decoder.layer0", model.gd.transformer.decoder.layers[0]),
                          ("decoder.layer5", model.gd.transformer.decoder.layers[5]),
                          ("dn_generator", model.dn_generator), ("mask_head", model.mask_head)):
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            finite = all(torch.isfinite(g).all().item() for g in gs)
            norm = float(sum(g.float().norm()**2 for g in gs)**0.5) if gs else 0.0
            grads[name] = {"finite": finite, "norm": round(norm, 6)}
            ok &= finite
        ok &= grads["decoder.layer0"]["norm"] > 0 and grads["decoder.layer5"]["norm"] > 0
        if dn is not None:
            ok &= grads["dn_generator"]["norm"] > 0
        r["grads"] = grads
        if device == "cuda":
            r["mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report.setdefault("train", []).append(r)
    model.zero_grad(set_to_none=True)

    # ---- eval: 不需要 targets; DN 开/关一致 ----
    model.eval()
    imgs, targets = next(iter(loader))
    imgs = [i.to(device) for i in imgs]
    caps = [t["caption"] for t in targets]
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model_nodn, _ = build_model(cfg["gdino_cfg"], None, cfg["bert_path"], device,
                                use_checkpoint=False, use_hed=True, use_dn=False)
    dropped = load_model_state(model_nodn, sd)
    model_nodn.eval()
    with torch.no_grad():
        a = model(imgs, caps)
        b = model_nodn(imgs, caps)
    keys = ("hed_all_logits", "hed_all_boxes", "hed_mask_embeds")
    # 两个独立构建的模型在 GPU 上可能选到不同的 kernel, 故报告最大差值并用极小容差判断
    max_diff = max((a[k].float() - b[k].float()).abs().nan_to_num(0.0).max().item() for k in keys)
    same = max_diff <= 1e-5
    report["eval"] = {
        "has_dn_outputs": "dn_outputs" in a,
        "dn_on_off_max_abs_diff": max_diff,
        "dn_on_off_identical": same,
        "dropped_keys_for_noDN": dropped
    }
    ok &= same and "dn_outputs" not in a
    del model_nodn

    # ---- 保存/加载往返 (内存, 不写盘) ----
    buf = io.BytesIO()
    torch.save({"model": sd, "cfg": cfg}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu")
    rc = resolve_model_mode({}, ck)
    # 模式与 dn 只从 checkpoint 的 cfg 恢复 (调用方配置为空), 检验恢复路径本身
    m2, _ = build_model(cfg["gdino_cfg"], None, cfg["bert_path"], device, use_checkpoint=False,
                        use_hed=rc["use_hed"], use_dn=rc["use_dn"], dn_cfg=rc["dn"])
    missing = load_model_state(m2, ck["model"])
    restored = (rc["use_hed"] and rc["use_dn"] and rc["dn"] == cfg["dn"] and not missing
                and torch.equal(m2.dn_generator.label_embedding.weight.cpu(),
                                sd["dn_generator.label_embedding.weight"].cpu())
                and m2.dn_reinit_rate == float(cfg["dn"]["reinit_rate"]))
    report["checkpoint_roundtrip"] = {"restored": bool(restored), "missing": missing}
    ok &= bool(restored)

    report["PASS"] = bool(ok)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
