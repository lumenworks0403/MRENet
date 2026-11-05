# HED 无 DN：继承 RefCOCO 基线，仅开启并行解码。
_base_ = "b0_refcoco.py"

use_hed = True
output_dir = 'work_dirs/b0_hed_refcoco_noDN'
