# 诊断配置：训练和验证使用同一批 RefCOCO 样本。
_base_ = "b0_refcoco.py"

train_data = {
    'type': 'refcoco_parquet',
    'path': 'data/refcoco/train-00000-of-00001-94431d5f4bd5b93f.parquet',
    'image_root': 'data',
    'max_samples': 16
}
val_data = {
    'type': 'refcoco_parquet',
    'path': 'data/refcoco/train-00000-of-00001-94431d5f4bd5b93f.parquet',
    'image_root': 'data',
    'max_samples': 16
}
short_side = 512
max_size = 853
hflip = 0.0
epochs = 40
lr_drop = 1000
num_workers = 0
output_dir = 'work_dirs/overfit'
eval_every = 5
N = 16
log_every = 4
