"""
数据集读取。统一的样本格式 (一条指代表达 = 一个样本, 可对应 0..N 个目标, 兼容 RefDrone 的多目标/无目标表达):
    {"image": 相对 image_root 的路径, "caption": str, "height": int, "width": int, "id": str,
     "objects": [{"bbox": [x, y, w, h] (像素, COCO 格式), "segmentation": COCO polygon list 或 RLE dict}, ...]}

支持:
  - RefCOCO/+/g: HuggingFace jxu124/refcoco* 的 parquet (含真实 COCO 分割多边形)
  - 通用 jsonl: 每行一个上述字典 (用于 RefDrone 等自建数据, 掩码可来自人工标注或 SAM 伪标注)
"""
import json
import random

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .preprocess import normalize_caption, resize_shape

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_refcoco_parquet(path, image_prefix_map=None):
    """jxu124/refcoco parquet -> 样本列表 (每个句子展开为一个样本)。
    image_path 形如 'coco/train2014/COCO_train2014_xxx.jpg'。"""
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    samples = []
    for r in rows:
        ann = json.loads(r["raw_anns"])
        info = json.loads(r["raw_image_info"])
        img = r["image_path"]
        if image_prefix_map:
            for a, b in image_prefix_map.items():
                img = img.replace(a, b)
        for s in r["sentences"]:
            samples.append({
                "image": img,
                "caption": s["sent"],
                "height": info["height"],
                "width": info["width"],
                "objects": [{
                    "bbox": ann["bbox"],
                    "segmentation": ann["segmentation"]
                }],
                "id": f"{r['ref_id']}_{s['sent_id']}",
            })
    return samples


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_samples(dcfg):
    if dcfg["type"] == "refcoco_parquet":
        s = load_refcoco_parquet(dcfg["path"])
    elif dcfg["type"] == "jsonl":
        s = load_jsonl(dcfg["path"])
    else:
        raise ValueError(dcfg["type"])
    if dcfg.get("max_samples"):
        s = s[:dcfg["max_samples"]]
    return s


def decode_mask(seg, h, w):
    if isinstance(seg, list):  # polygons
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):  # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


class RefSegDataset(Dataset):
    """短边缩放到 short_side (长边不超过 max_size), 训练时可水平翻转。
    注意: 水平翻转会让 'left/right' 描述失效, 因此翻转时同步交换 caption 中的 left/right。"""

    def __init__(self, samples, image_root, train=True, short_side=800, max_size=1333, hflip=0.5):
        self.samples = samples
        self.image_root = image_root
        self.train = train
        self.short_side = short_side
        self.max_size = max_size
        self.hflip = hflip if train else 0.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(f"{self.image_root}/{s['image']}").convert("RGB")
        w0, h0 = img.size
        objs = s["objects"]
        masks = np.stack([decode_mask(o["segmentation"], h0, w0) for o in objs]) if objs \
            else np.zeros((0, h0, w0), np.uint8)
        boxes = np.array([o["bbox"] for o in objs], dtype=np.float32).reshape(-1, 4)
        caption = s["caption"]

        if self.hflip and random.random() < self.hflip:
            img = TF.hflip(img)
            masks = masks[:, :, ::-1].copy()
            boxes[:, 0] = w0 - boxes[:, 0] - boxes[:, 2]
            caption = (caption.replace("left", "\0").replace("right",
                                                             "left").replace("\0", "right"))

        h, w = resize_shape(h0, w0, self.short_side, self.max_size)
        img_t = TF.normalize(TF.to_tensor(TF.resize(img, [h, w])), IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(masks).float()
        if len(mask_t):
            mask_t = torch.nn.functional.interpolate(mask_t[None], size=(h, w), mode="nearest")[0]
        else:
            mask_t = mask_t.new_zeros((0, h, w))
        b = torch.from_numpy(boxes)
        box = torch.stack([(b[:, 0] + b[:, 2] / 2) / w0,
                           (b[:, 1] + b[:, 3] / 2) / h0, b[:, 2] / w0, b[:, 3] / h0], 1)

        caption = normalize_caption(caption)
        target = {
            "caption": caption,
            "boxes": box,  # (n, 4) cxcywh, 归一化; n 可为 0
            "masks": mask_t,  # (n, h, w) 缩放后尺寸
            "orig_mask":
            torch.from_numpy(masks.any(0) if len(masks) else np.zeros((h0,
                                                                       w0), bool)),  # 所有目标并集, 评估用
            "orig_boxes": torch.from_numpy(boxes),  # (n, 4) xywh 原图像素 (翻转后), 评估用
            "size": torch.tensor([h, w]),
            "orig_size": torch.tensor([h0, w0]),
            "id": s["id"],
        }
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
