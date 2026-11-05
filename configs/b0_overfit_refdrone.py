# 诊断配置：使用框填充掩码检查 RefDrone 多目标流程，不代表分割精度。
_base_ = "b0_refdrone.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/refdrone/test51_boxmask.jsonl',
    'image_root': 'data',
    'max_samples': None
}
val_data = {
    'type': 'jsonl',
    'path': 'data/refdrone/test51_boxmask.jsonl',
    'image_root': 'data',
    'max_samples': None
}
short_side = 512
max_size = 853
hflip = 0.0
epochs = 40
lr_drop = 1000
num_workers = 0
output_dir = 'work_dirs/overfit_refdrone'
eval_every = 5
log_every = 10
