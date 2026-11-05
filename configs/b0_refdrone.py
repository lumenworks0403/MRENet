# RefDrone 上的 B0。RefDrone 为多目标 / 无目标指代, 掩码需先用 tools/convert_refdrone.py 生成 (推荐 SAM 伪掩码)。
_base_ = "b0_refcoco.py"

train_data = {
    'type': 'jsonl',
    'path': 'data/refdrone/train.jsonl',
    'image_root': 'data',
    'max_samples': None
}
val_data = {
    'type': 'jsonl',
    'path': 'data/refdrone/val.jsonl',
    'image_root': 'data',
    'max_samples': None
}
output_dir = 'work_dirs/b0_refdrone'
postprocess = {'mode': 'multi', 'score_thr': 0.3, 'max_dets': 50}
