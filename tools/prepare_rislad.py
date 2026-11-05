"""Convert official RIS-LAD annotations to B0 JSONL; preserve official splits."""
import argparse
import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu


class DataOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/RIS-LAD')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    data = json.loads((root / 'ris_lad/instances.json').read_text(encoding='utf-8'))
    refs = DataOnlyUnpickler(io.BytesIO((root / 'ris_lad/refs(unc).p').read_bytes())).load()
    images = {x['id']: x for x in data['images']}
    anns = {x['id']: x for x in data['annotations']}
    groups = {s: [] for s in ('train', 'val', 'test')}
    split_images = {s: set() for s in groups}
    checked_images = set()
    seen_ids = set()
    for ref in refs:
        split = ref['split']
        if split not in groups:
            raise ValueError(f'Unknown split: {split}')
        im = images[ref['image_id']]
        ann = anns[ref['ann_id']]
        assert ann['image_id'] == im['id']
        h, w = im['height'], im['width']
        rel = Path('images/ris_lad') / im['file_name']
        path = (root / rel).resolve()
        if root not in path.parents:
            raise ValueError('Image path outside dataset root')
        if im['id'] not in checked_images:
            with Image.open(path) as image:
                assert image.size == (w, h), str(path)
                image.verify()
            checked_images.add(im['id'])
        seg = ann['segmentation']
        if isinstance(seg, list) and seg and isinstance(seg[0], dict):
            mask = mu.decode(seg)
        elif isinstance(seg, list):
            mask = mu.decode(mu.merge(mu.frPyObjects(seg, h, w)))
        elif isinstance(seg['counts'], list):
            mask = mu.decode(mu.frPyObjects(seg, h, w))
        else:
            mask = mu.decode(seg)
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        assert mask.shape == (h, w) and mask.any(), f'Invalid mask: {ann["id"]}'
        # Canonical single RLE dictionary supported by existing B0 decoder.
        rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        bbox = list(ann['bbox'])
        x, y, bw, bh = bbox
        assert bw > 0 and bh > 0 and x >= 0 and y >= 0 and x + bw <= w + 1 and y + bh <= h + 1
        for sent in ref['sentences']:
            uid = f'{ref["ref_id"]}_{sent["sent_id"]}'
            assert uid not in seen_ids
            seen_ids.add(uid)
            caption = sent.get('sent', sent.get('raw', '')).strip()
            assert caption
            groups[split].append(
                dict(id=uid, image=rel.as_posix(), caption=caption, height=h, width=w,
                     objects=[dict(bbox=bbox, segmentation=rle)]))
        split_images[split].add(im['id'])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not split_images[a] & split_images[b], f'Overlapping images: {a}/{b}'
    out = root / 'b0'
    out.mkdir(exist_ok=True)
    for split, rows in groups.items():
        path = out / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {s: dict(expressions=len(groups[s]), images=len(split_images[s])) for s in groups}
    report['checks'] = 'all referenced images verified; all masks decoded; no cross-split image IDs'
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
