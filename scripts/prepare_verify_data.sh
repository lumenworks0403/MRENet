#!/usr/bin/env bash
# 准备验收用的小数据 (与本地实测完全相同的样本):
#   - RefCOCO 标注 parquet + 训练/验证子集所需的 COCO 图片 (约 40 张)
#   - RefDrone test 前 51 条表达 + 对应的 26 张 VisDrone 原图 (从 HF 的 base64 tsv 头部截取)
#   bash scripts/prepare_verify_data.sh
set -e
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
# HF 镜像和 COCO 图片站都不需要 AutoDL 学术加速代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p data/refcoco data/refdrone/tsv_head data/VisDrone2019/all_image
HF=${HF_ENDPOINT:-https://hf-mirror.com}/datasets
for s in validation-00000-of-00001-bfeafdc84ca37aa2 train-00000-of-00001-94431d5f4bd5b93f; do
  [ -s data/refcoco/$s.parquet ] || wget -q -O data/refcoco/$s.parquet $HF/jxu124/refcoco/resolve/main/data/$s.parquet
done
python tools/fetch_coco_images.py --parquet data/refcoco/train-00000-of-00001-94431d5f4bd5b93f.parquet --max-samples 200
python tools/fetch_coco_images.py --parquet data/refcoco/validation-00000-of-00001-bfeafdc84ca37aa2.parquet --max-samples 100

for s in train val test; do
  [ -s data/refdrone/RefDrone_${s}_mdetr.json ] || wget -q -O data/refdrone/RefDrone_${s}_mdetr.json $HF/sunzc-sunny/RefDrone/resolve/main/RefDrone_${s}_mdetr.json
done
[ -s data/refdrone/tsv_head/head.tsv ] || curl -sSL -r 0-12000000 -o data/refdrone/tsv_head/head.tsv \
  $HF/sunzc-sunny/RefDrone/resolve/main/refdrone_test_base64.tsv
python - <<'EOF'
import base64, io, json, os
from PIL import Image
d = json.load(open("data/refdrone/RefDrone_test_mdetr.json"))
lines = open("data/refdrone/tsv_head/head.tsv", encoding="utf-8").read().split("\n")[1:-1]  # 最后一行被 Range 截断
keep = set()
for l in lines:
    r = l.split("\t"); im = d["images"][int(r[0])]
    p = "data/VisDrone2019/all_image/" + im["file_name"]
    if not os.path.exists(p):
        b = base64.b64decode(r[5]); Image.open(io.BytesIO(b)).verify(); open(p, "wb").write(b)
    keep.add(im["id"])
d["images"] = [i for i in d["images"] if i["id"] in keep]
d["annotations"] = [a for a in d["annotations"] if a["image_id"] in keep]
json.dump(d, open("data/refdrone/RefDrone_test51_mdetr.json", "w"))
print(len(d["images"]), "expressions,", len({i["file_name"] for i in d["images"]}), "images")
EOF
python tools/convert_refdrone.py --ann data/refdrone/RefDrone_test51_mdetr.json \
  --image-dir VisDrone2019/all_image --out data/refdrone/test51_boxmask.jsonl --mask-source box
