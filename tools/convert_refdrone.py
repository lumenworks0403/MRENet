"""RefDrone (MDETR json) -> B0 jsonl。

RefDrone 只提供框, 没有分割掩码。--mask-source 选项:
  sam : 以 GT 框为提示, 用 SAM 生成伪掩码 (推荐; 需 pip install segment-anything 并下载 SAM 权重)。
        生成后应人工抽检, 论文中需说明掩码来源。
  box : 用框内全部像素作掩码。仅用于跑通流程/调试, 不能作为指代分割结果报告。

    python tools/convert_refdrone.py --ann data/refdrone/RefDrone_train_mdetr.json \
        --image-dir VisDrone2019/all_image --out data/refdrone/train.jsonl --mask-source sam \
        --sam-ckpt weights/sam_vit_h_4b8939.pth --image-root data
"""
import argparse
import collections
import json
import os

import numpy as np
from pycocotools import mask as mask_utils


def box_mask_rle(bbox, h, w):
    m = np.zeros((h, w), np.uint8, order="F")
    x, y, bw, bh = [int(round(v)) for v in bbox]
    m[max(y, 0):y + bh, max(x, 0):x + bw] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode()
    return rle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--image-dir", required=True, help="相对 --image-root 的图片目录")
    ap.add_argument("--image-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-source", choices=["sam", "box"], required=True)
    ap.add_argument("--sam-ckpt", default=None)
    ap.add_argument("--sam-type", default="vit_h")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    d = json.load(open(args.ann, encoding="utf-8"))
    anns = collections.defaultdict(list)
    for a in d["annotations"]:
        if not a.get("empty", False) and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            anns[a["image_id"]].append(a)
    images = d["images"][:args.max_samples] if args.max_samples else d["images"]

    predictor = None
    if args.mask_source == "sam":
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[args.sam_type](
            checkpoint=args.sam_ckpt).to("cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(sam)

    stats = collections.Counter()
    last_file = None
    with open(args.out, "w", encoding="utf-8") as f:
        for im in images:
            h, w = im["height"], im["width"]
            rel = f"{args.image_dir}/{im['file_name']}"
            objs = []
            boxes = [a["bbox"] for a in anns.get(im["id"], [])]
            if boxes and predictor is not None:
                if last_file != rel:
                    from PIL import Image
                    predictor.set_image(
                        np.array(Image.open(os.path.join(args.image_root, rel)).convert("RGB")))
                    last_file = rel
                import torch
                xyxy = torch.tensor([[x, y, x + bw, y + bh] for x, y, bw, bh in boxes],
                                    device=predictor.device)
                tb = predictor.transform.apply_boxes_torch(xyxy, (h, w))
                masks, _, _ = predictor.predict_torch(None, None, boxes=tb, multimask_output=False)
                for b, m in zip(boxes, masks[:, 0].cpu().numpy()):
                    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode()
                    objs.append({"bbox": b, "segmentation": rle})
            else:
                objs = [{"bbox": b, "segmentation": box_mask_rle(b, h, w)} for b in boxes]
            stats[min(len(objs), 5)] += 1
            f.write(
                json.dumps(
                    {
                        "image": rel,
                        "caption": im["caption"],
                        "height": h,
                        "width": w,
                        "objects": objs,
                        "id": str(im["id"]),
                        "mask_source": args.mask_source
                    }, ensure_ascii=False) + "\n")
    print(
        f"{len(images)} expressions -> {args.out}; targets per expression (5 = 5+): {dict(sorted(stats.items()))}"
    )


if __name__ == "__main__":
    main()
