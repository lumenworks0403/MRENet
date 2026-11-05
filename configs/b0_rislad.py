# Run from B0 root. Preserve official RIS-LAD splits.
_base_ = "b0_refcoco.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/RIS-LAD/b0/train.jsonl',
    'image_root': 'data/RIS-LAD',
    'max_samples': None
}
val_data = {
    'type': 'jsonl',
    'path': 'data/RIS-LAD/b0/val.jsonl',
    'image_root': 'data/RIS-LAD',
    'max_samples': None
}
hflip = 0.0
batch_size = 1
output_dir = 'work_dirs/b0_rislad'
