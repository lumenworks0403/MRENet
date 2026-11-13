_base_ = "mrenet_rislad.py"
adaptation_only = True  # 使用 tools/finetune.py 的两阶段流程。
train_data = dict(type="jsonl", path="data/RRSIS-D/b0/train.jsonl", image_root="data/RRSIS-D")
val_data = dict(type="jsonl", path="data/RRSIS-D/b0/val.jsonl", image_root="data/RRSIS-D")
test_data = dict(type="jsonl", path="data/RRSIS-D/b0/test.jsonl", image_root="data/RRSIS-D")
fewshot_dir = "data/RRSIS-D/b0/fewshot"
batch_size = 2
grad_accum = 1
hflip = 0.0
output_dir = "work_dirs/mrenet_rrsisd"
