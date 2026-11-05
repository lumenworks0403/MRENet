# B0_HED + MMSCopE + RefScoreHead，无 DN — RRSIS-D 清理版。
_base_ = "b0_hed_mm_score_rislad.py"

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
output_dir = 'work_dirs/b0_hed_mm_score_rrsisd'
