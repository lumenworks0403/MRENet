# B0 — RRSIS-D 清理版，使用排除异常记录后的 JSONL。
_base_ = "b0_rislad.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/RRSIS-D/b0/train.jsonl',
    'image_root': 'data/RRSIS-D',
    'max_samples': None
}
val_data = {
    'type': 'jsonl',
    'path': 'data/RRSIS-D/b0/val.jsonl',
    'image_root': 'data/RRSIS-D',
    'max_samples': None
}
output_dir = 'work_dirs/b0_rrsisd'
