"""单张图片推理: 输出掩码 PNG 与可视化。

    python tools/infer.py --ckpt work_dirs/b0_refcoco/best.pth --image xxx.jpg --text "the white car on the left" --out vis.jpg
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from rrs.preprocess import normalize_caption, resize_shape  # noqa: E402
from rrs.evaluate import postprocess_for_model  # noqa: E402
from rrs.model import build_model, load_model_state  # noqa: E402
from rrs.model_mode import model_variant, resolve_model_mode  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="vis.jpg")
    ap.add_argument("--short-side", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1333)
    ap.add_argument("--hed-mask-mode", choices=("avg", "last"), default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = resolve_model_mode(ck["cfg"], ck)
    if args.hed_mask_mode is not None:
        cfg["hed_mask_mode"] = args.hed_mask_mode
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(os.path.join(ROOT, cfg["gdino_cfg"]), None,
                           os.path.join(ROOT, cfg["bert_path"]), dev, use_checkpoint=False,
                           use_hed=cfg["use_hed"], use_dn=cfg["use_dn"], dn_cfg=cfg["dn"],
                           use_mmscope=cfg["use_mmscope"], mm_cfg=cfg["mm"],
                           use_ref_score=cfg["use_ref_score"])
    load_model_state(model, ck["model"])
    model.eval()  # eval 前向不生成 DN, 不需要 targets
    print({
        "use_hed": cfg["use_hed"],
        "hed_mask_mode": cfg["hed_mask_mode"],
        "use_dn": cfg["use_dn"],
        "use_mmscope": cfg["use_mmscope"],
        "use_ref_score": cfg["use_ref_score"],
        "model_variant": model_variant(cfg)
    })

    img = Image.open(args.image).convert("RGB")
    w0, h0 = img.size
    h, w = resize_shape(h0, w0, args.short_side, args.max_size)
    x = TF.normalize(TF.to_tensor(TF.resize(img, [h, w])), IMAGENET_MEAN, IMAGENET_STD).to(dev)
    caption = normalize_caption(args.text)
    with torch.no_grad():
        out = model([x], [caption])
    r = postprocess_for_model(model, out, [{
        "size": torch.tensor([h, w]),
        "orig_size": torch.tensor([h0, w0])
    }], cfg.get("postprocess", {}), cfg["hed_mask_mode"])[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mask = r["mask"].cpu().numpy()
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.splitext(args.out)[0] + "_mask.png")
    vis = np.array(img).astype(np.float32)
    vis[mask] = vis[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    Image.fromarray(vis.astype(np.uint8)).save(args.out)
    print({
        "num_targets": len(r["queries"]),
        "scores": [round(v, 3) for v in r["scores"]],
        # score_type 标明 scores 的含义 ("ref_score" = 新评分头, "token" = 原 token 均分);
        # 新模式下 token_scores 只是诊断值, 不参与选择。
        "score_type": r.get("score_type", "token"),
        "token_scores": [round(v, 3) for v in r.get("token_scores", [])],
        "queries": r["queries"],
        "boxes_xyxy": [[round(v, 1) for v in b] for b in r["boxes"]],
        "mask_pixels": int(mask.sum())
    })


if __name__ == "__main__":
    main()
