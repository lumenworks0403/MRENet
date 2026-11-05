"""按标注只下载需要的 COCO train2014 图片 (RefCOCO 全部图片均来自 train2014)。

    python tools/fetch_coco_images.py --parquet data/refcoco/xxx.parquet --max-samples 200 --out data
完整训练请直接下载 http://images.cocodataset.org/zips/train2014.zip (13GB) 并解压到 data/coco/train2014。
"""
import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rrs.data import load_refcoco_parquet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    samples = load_refcoco_parquet(args.parquet)
    if args.max_samples:
        samples = samples[:args.max_samples]
    paths = sorted({s["image"] for s in samples})

    def fetch(p):
        dst = os.path.join(args.out, p)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = "http://images.cocodataset.org/" + p.split("coco/", 1)[1]
        urllib.request.urlretrieve(url, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        return 1

    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(fetch, paths))
    print(f"{len(samples)} samples, {len(paths)} images, downloaded {n}")


if __name__ == "__main__":
    main()
