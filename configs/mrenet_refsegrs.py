_base_ = "mrenet_rislad.py"
adaptation_only = True  # 使用 tools/finetune.py 的两阶段流程。
train_data = dict(type="jsonl", path="data/RefSegRS/b0/train.jsonl", image_root="data/RefSegRS")
val_data = dict(type="jsonl", path="data/RefSegRS/b0/val.jsonl", image_root="data/RefSegRS")
test_data = dict(type="jsonl", path="data/RefSegRS/b0/test.jsonl", image_root="data/RefSegRS")
fewshot_dir = "data/RefSegRS/b0/fewshot"
batch_size = 2
grad_accum = 1
hflip = 0.0
output_dir = "work_dirs/mrenet_refsegrs"
