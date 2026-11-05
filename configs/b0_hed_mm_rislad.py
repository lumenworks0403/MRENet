# HED + MMSCopE：RIS-LAD；需安装真实 SS2D 算子。
_base_ = "b0_hed_rislad.py"

output_dir = 'work_dirs/b0_hed_mm_rislad'
use_mmscope = True
mm = {
    'channels': 128,
    'feat_proj_dim': 256,
    'norm': 'gn',
    'gn_groups': 32,
    'scan_fp32': True,
    'drop_path': 0.0
}
