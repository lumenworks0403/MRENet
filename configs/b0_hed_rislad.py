# HED 无 DN：RIS-LAD，batch_size=2、grad_accum=2。
# 对比实验需统一实际 batch_size；padding 画布会影响掩码插值。
_base_ = "b0_rislad.py"

batch_size = 2
grad_accum = 2
output_dir = 'work_dirs/b0_hed_rislad_noDN'
use_hed = True
