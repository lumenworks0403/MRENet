# B0 — RefSegRS，使用转换后的 JSONL，保留官方划分。
_base_ = "b0_rislad.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/RefSegRS/b0/train.jsonl',
    'image_root': 'data/RefSegRS',
    'max_samples': None
}
val_data = {
    'type': 'jsonl',
    'path': 'data/RefSegRS/b0/val.jsonl',
    'image_root': 'data/RefSegRS',
    'max_samples': None
}
output_dir = 'work_dirs/b0_refsegrs'
