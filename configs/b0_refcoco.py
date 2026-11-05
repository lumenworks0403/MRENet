# B0 训练配置。模型结构超参来自官方 groundingdino/config/GroundingDINO_SwinT_OGC.py (见 gdino_cfg),
# 训练超参参考 Open-GroundingDino config/cfg_odvg.py。
gdino_cfg = "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
gdino_ckpt = "weights/groundingdino_swint_ogc.pth"
bert_path = "weights/bert-base-uncased"

# 数据: type = "refcoco_parquet" | "jsonl"
train_data = dict(type="refcoco_parquet",
                  path="data/refcoco/train-00000-of-00001-94431d5f4bd5b93f.parquet",
                  image_root="data", max_samples=None)
val_data = dict(type="refcoco_parquet",
                path="data/refcoco/validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
                image_root="data", max_samples=None)
short_side = 800
max_size = 1333
hflip = 0.5

# 优化 (Open-GroundingDino: lr=1e-4, lr_backbone=1e-5, freeze bert, weight_decay=1e-4, clip 0.1)
lr = 1e-4
lr_backbone = 1e-5
lr_backbone_names = ["gd.backbone.0"]
freeze_keywords = ["gd.bert"]
weight_decay = 1e-4
clip_max_norm = 0.1
epochs = 12
lr_drop = 8
batch_size = 2
grad_accum = 1
num_workers = 2
amp = True
use_checkpoint = True  # Swin / encoder 梯度检查点, 省显存

# 匹配与损失权重 (cls/bbox/giou 同 Open-GroundingDino; mask/dice 同 Mask2Former)
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
set_cost_mask = 5.0
set_cost_dice = 5.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
mask_loss_coef = 5.0
dice_loss_coef = 5.0
num_points = 12544

output_dir = "work_dirs/b0_refcoco"
eval_every = 1
seed = 42

# 推理后处理: RefCOCO 为单目标指代
postprocess = dict(mode="top1")
