# 论文完整模型，第 3 节及第 4.2.1 节。
_base_ = "b0_rislad.py"
use_sped = True
use_mcpe = True
use_rca = True
use_dn = True
mask_aux_layers = True
sped_mask_mode = "avg"
mcpe = dict(channels=128, feat_proj_dim=256, norm="gn", gn_groups=32, scan_fp32=True, drop_path=0.0)
dn = dict(label_book_size=1, label_noise_scale=0.5, box_noise_scale=1.0, num_dn_queries=100,
          reinit_rate=0.5, loss_coef=1.0, branch_ref_mode="sigmoid")
ref_score_loss_coef = 1.0
batch_size = 4
grad_accum = 1
epochs = 50
lr_drop = 30
hflip = 0.0
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
postprocess = dict(mode="top1")
output_dir = "work_dirs/mrenet_rislad"
