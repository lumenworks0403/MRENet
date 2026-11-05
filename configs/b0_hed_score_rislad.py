# HED + RefScoreHead：RIS-LAD，使用 FPN 像素支路。
_base_ = "b0_hed_rislad.py"

output_dir = 'work_dirs/b0_hed_score_rislad'
use_ref_score = True
ref_score_loss_coef = 1.0
