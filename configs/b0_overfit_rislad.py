# 诊断配置：训练和验证使用同一批 RIS-LAD 样本。
_base_ = "b0_rislad.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/RIS-LAD/b0/train.jsonl',
    'image_root': 'data/RIS-LAD',
    'max_samples': 16
}
val_data = {
    'type': 'jsonl',
    'path': 'data/RIS-LAD/b0/train.jsonl',
    'image_root': 'data/RIS-LAD',
    'max_samples': 16
}
short_side = 512
max_size = 853
epochs = 40
lr_drop = 1000
num_workers = 0
output_dir = 'work_dirs/overfit_rislad'
eval_every = 5
log_every = 4
