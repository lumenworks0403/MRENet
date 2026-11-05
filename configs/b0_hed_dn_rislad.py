# B0_HED  HED+DN — RIS-LAD 配置。
_base_ = "b0_hed_rislad.py"

output_dir = 'work_dirs/b0_hed_dn_rislad'
use_dn = True
dn = {
    'label_book_size': 1,
    'label_noise_scale': 0.5,
    'box_noise_scale': 1.0,
    'num_dn_queries': 100,
    'reinit_rate': 0.5,
    'loss_coef': 1.0,
    'branch_ref_mode': 'sigmoid'
}
