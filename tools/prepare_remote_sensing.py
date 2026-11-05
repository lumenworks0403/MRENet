"""将 RefSegRS、RRSIS-D 转为本项目 JSONL；仅转换数据，不运行模型或测试。

    python tools/prepare_remote_sensing.py
    python tools/prepare_remote_sensing.py --dataset rrsisd

原始文件保持不变。生成文件写到各数据集的 b0/ 下，重复运行会覆盖这些生成文件。
RRSIS-D 的空掩码、尺寸不匹配样本从输出划分排除，完整记录在 excluded_samples.jsonl。
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from pycocotools import mask as mu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')


class DataOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def make_object(mask):
    """保留整张语义掩码，并使用其紧致 xywh 外接框。"""
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = mu.toBbox(rle).tolist()
    rle['counts'] = rle['counts'].decode('ascii')
    return dict(bbox=bbox, segmentation=rle)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_bundle(root, dataset, groups, source_counts, exclusions, sources):
    out = root / 'b0'
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(out / f'{split}.jsonl', groups[split])
    write_jsonl(out / 'excluded_samples.jsonl', exclusions)
    excluded_counts = Counter(row['split'] for row in exclusions)
    reasons = Counter(reason for row in exclusions for reason in row['reasons'])
    report = dict(
        dataset=dataset,
        generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        format='B0 JSONL: one expression, one full binary mask, COCO xywh bbox',
        sources=sources,
        splits={
            split:
            dict(
                source_expressions=source_counts[split],
                output_expressions=len(groups[split]),
                excluded_expressions=excluded_counts[split],
                output_images=len({row['image']
                                   for row in groups[split]}),
            )
            for split in SPLITS
        },
        output_expressions=sum(len(rows) for rows in groups.values()),
        excluded_expressions=len(exclusions),
        exclusion_reasons=dict(reasons),
        bbox_policy='recomputed from binary mask with pycocotools.mask.toBbox; pixel xywh',
        mask_policy='preserve full foreground; encode one compressed COCO RLE dictionary',
        anomaly_policy=
        'exclude empty or size-mismatched masks; retain source files and exclusion records',
        split_policy='preserve original split membership; never move samples between splits',
        evaluation_scope=('cleaned subset; not the full official validation/test split'
                          if exclusions else 'all official samples retained'),
        source_files_modified=False,
        tests_run=False,
        training_run=False,
    )
    report_path = out / 'conversion_report.json'
    temporary = report_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved: {out}', flush=True)


def prepare_refsegrs(root):
    groups = {split: [] for split in SPLITS}
    source_counts = Counter()
    seen = set()
    for split in SPLITS:
        text_path = root / f'output_phrase_{split}.txt'
        for line_number, line in enumerate(
                text_path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            image_id, caption = line.strip().split(maxsplit=1)
            if not image_id.isdigit() or image_id in seen:
                raise ValueError(f'Invalid or duplicate ID: {text_path}:{line_number}')
            seen.add(image_id)
            source_counts[split] += 1
            relative_image = f'images/{image_id}.tif'
            with Image.open(root / relative_image) as image:
                width, height = image.size
            with Image.open(root / 'masks' / f'{image_id}.tif') as image:
                pixels = np.array(image)
            # 原始 RefSegRS 是三个通道相等的黑白 RGB 掩码。
            mask = (pixels[:, :, 0] if pixels.ndim == 3 else pixels) > 0
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f'Unexpected RefSegRS mask: {image_id}, shape={mask.shape}')
            groups[split].append(
                dict(
                    image=relative_image,
                    caption=caption,
                    height=height,
                    width=width,
                    id=f'refsegrs_{image_id}',
                    objects=[make_object(mask)],
                ))
            if source_counts[split] % 500 == 0:
                print(f'RefSegRS {split}: {source_counts[split]} converted', flush=True)
    write_bundle(root, 'RefSegRS', groups, source_counts, [],
                 [f'output_phrase_{split}.txt' for split in SPLITS] + ['images/', 'masks/'])


def decode_annotation(segmentation, height, width):
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        if isinstance(segmentation[0], dict):
            mask = mu.decode(segmentation)
        else:
            mask = mu.decode(mu.merge(mu.frPyObjects(segmentation, height, width)))
    elif isinstance(segmentation['counts'], list):
        mask = mu.decode(mu.frPyObjects(segmentation, height, width))
    else:
        mask = mu.decode(segmentation)
    return mask.any(axis=2) if mask.ndim == 3 else mask > 0


def prepare_rrsisd(root):
    with (root / 'rrsisd/instances.json').open(encoding='utf-8') as f:
        instances = json.load(f)
    refs = DataOnlyUnpickler(io.BytesIO((root / 'rrsisd/refs(unc).p').read_bytes())).load()
    images = {row['id']: row for row in instances['images']}
    annotations = {row['id']: row for row in instances['annotations']}
    groups = {split: [] for split in SPLITS}
    source_counts = Counter()
    exclusions = []
    seen = set()
    for index, ref in enumerate(refs, 1):
        split = ref['split']
        if split not in groups:
            raise ValueError(f'Unknown split: {split}')
        im = images[ref['image_id']]
        ann = annotations[ref['ann_id']]
        if ann['image_id'] != im['id']:
            raise ValueError(f'Image/annotation mismatch: {ref["ref_id"]}')
        relative_image = (Path('images/rrsisd/JPEGImages') / im['file_name']).as_posix()
        path = (root / relative_image).resolve()
        if root not in path.parents:
            raise ValueError(f'Image path outside dataset root: {relative_image}')
        # 读取图像头以写入真实尺寸；不做图像测试或模型运行。
        with Image.open(path) as image:
            width, height = image.size
        mask = decode_annotation(ann['segmentation'], im['height'], im['width'])
        reasons = []
        if mask.shape != (height, width):
            reasons.append('mask_image_size_mismatch')
        if not mask.any():
            reasons.append('empty_mask_with_positive_expression')
        if (im['height'], im['width']) != (height, width):
            reasons.append('annotation_image_size_mismatch')
        obj = None if reasons else make_object(mask)
        for sent in ref['sentences']:
            uid = f'rrsisd_{ref["ref_id"]}_{sent["sent_id"]}'
            caption = sent.get('sent', sent.get('raw', '')).strip()
            if not caption or uid in seen:
                raise ValueError(f'Empty caption or duplicate ID: {uid}')
            seen.add(uid)
            source_counts[split] += 1
            if reasons:
                exclusions.append(
                    dict(
                        id=uid,
                        split=split,
                        image=relative_image,
                        caption=caption,
                        reasons=reasons,
                        action='excluded_from_training_jsonl_source_retained',
                        image_size_hw=[height, width],
                        mask_size_hw=list(mask.shape),
                        source_image=im,
                        source_annotation=ann,
                        source_reference=ref,
                    ))
                continue
            groups[split].append(
                dict(
                    image=relative_image,
                    caption=caption,
                    height=height,
                    width=width,
                    id=uid,
                    objects=[obj],
                ))
        if index % 2000 == 0:
            print(f'RRSIS-D: {index}/{len(refs)} references processed', flush=True)
    write_bundle(root, 'RRSIS-D', groups, source_counts, exclusions,
                 ['rrsisd/instances.json', 'rrsisd/refs(unc).p', 'images/rrsisd/JPEGImages/'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'refsegrs', 'rrsisd'], default='all')
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if args.dataset in ('all', 'refsegrs'):
        prepare_refsegrs(data_root / 'RefSegRS')
    if args.dataset in ('all', 'rrsisd'):
        prepare_rrsisd(data_root / 'RRSIS-D')


if __name__ == '__main__':
    main()
