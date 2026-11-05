# HED + MMSCopE + RefScoreHead：RIS-LAD，默认关闭 DN。
_base_ = "b0_hed_mm_rislad.py"

output_dir = 'work_dirs/b0_hed_mm_score_rislad'
use_ref_score = True
ref_score_loss_coef = 1.0
