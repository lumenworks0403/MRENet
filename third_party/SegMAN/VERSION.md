# SegMAN 官方源码来源与版本

MMSCopE 像素支路所需的官方实现。只取 MMSCopE 的计算路径，**不引入** SegMAN 编码器、
LASS、natten、MMSeg 训练框架或其注册器。

## 来源

- 仓库：<https://github.com/yunxiangfu2001/SegMAN>
- 获取方式：`git clone --depth 1`（2026-10-16，默认分支 main）
- **commit：`9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8`**（提交日期 2026-06-03）
- 论文：SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for
  Semantic Segmentation (CVPR 2025)，<https://arxiv.org/abs/2412.11890>；
  MMSCopE 的算法说明见第 3.3 节与图 3(d)，本地 PDF 为 `../../../SegMAN.pdf`。
- 许可：Apache-2.0，见本目录 `LICENSE`。被抽取代码里的 NVIDIA 版权声明随代码保留在
  `segman/mmscope_core.py` 文件头。

`main` 只是定位入口，上面的 commit 才是本次固定的版本。

## 文件清单与 SHA256

| 本目录文件 | 来源路径（上述 commit） | SHA256 | 是否改动 |
|---|---|---|---|
| `LICENSE` | `LICENSE` | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` | 原样 |
| `segman/csm_triton.py` | `segmentation/mmseg/models/decode_heads/csm_triton.py` | `0d78812b26d3e06e44230b4d5fad6dfc4e089b8e268b5db6c19b265171f07a68` | 原样 |
| `segman/segman_decoder_official.py` | `segmentation/mmseg/models/decode_heads/segman_decoder.py` | `a163365fc0bd1a43b09ba4bbecbdd706b97b7bc775e32671be6225db50a095b5` | 原样（**只作对照，不被导入**） |
| `kernels/selective_scan/` | `kernels/selective_scan/`（去掉 `.ipynb_checkpoints`） | 目录，见下 | 原样 |
| `segman/mmscope_core.py` | 从 `segman_decoder.py` 抽取 | 新文件 | 见文件头「与官方的差异」 |

`kernels/selective_scan` 是 VMamba 的 selective scan CUDA 扩展（SegMAN README 指向
<https://github.com/MzeroMiko/VMamba>），编译出 `selective_scan_cuda_oflex`。目录原样复制，
未改动任何 `.cu/.cuh/.cpp/setup.py`。

参考配置（未复制，仅用于确定内部通道宽度的依据）：
`segmentation/local_configs/segman/{tiny,small,base}/segman_*_ade.py`，
其中 `segman_s_ade.py` 的 SHA256 为
`1885a62dfb5a2f24f7fe25c5506015854b1308880b5b56a32faf88f28b02cde9`。

## 抽取到 `segman/mmscope_core.py` 的类

| 类 | 官方位置（`segman_decoder.py` 行号） | 用途 | 去注释/空白后与官方逐字符比对 |
|---|---|---|---|
| `SelectiveScanOflex` | L366–384 | `selective_scan_cuda_oflex` 的 autograd 封装 | 一致（720 字符） |
| `LayerScale` | L430–449 | VSSMBlock 的可选 layer scale（默认 `layerscale=False`，走 Identity） | 一致（531 字符） |
| `LayerNorm2d` | L461–469 | VSSMBlock 的 norm | 一致（199 字符） |
| `VSSM` | L484–655 | **SS2D 本体**，MMSCopE 的真实扫描路径 | 一致（3807 字符，排除延迟导入的 `require_scan_ops()` 一行） |
| `FFN` | L737–762 | VSSMBlock 的 channel mixer | 一致（482 字符） |
| `VSSMBlock` | L765–814 | `SegMANDecoder.forward_winssm` 里的 `self.vssm` | 一致（1277 字符） |

比对方法（2026-10-16 在本机用 shell 文本工具执行，**未运行 Python**）：
从官方副本 `segman/segman_decoder_official.py` 与本文件按上表行号各取一段，
逐行去掉 `#` 注释、空行与 docstring，再删除全部空白字符后用 `cmp` 整体比对。
这证明的是**源码文本一致**（除注释、PEP8 空格、`VSSM.__init__` 里多出的
`require_scan_ops()` 一行），**不是**运行时数值等价——数值层面的检查见
`tools/check_ss2d.py`（待服务器执行）与
`docs/MMSCopE与评分头接入说明_20261016.md` 第 8.1 节的 L1/L2/L3 分层。

未抽取（MMSCopE 路径不调用）：`RoPE`、`Attention`（natten）、`EfficientScan`/`EfficientMerge`、
`cross_scan_continuous`/`get_continuous_paths`、`MLP`、`SegMANDecoder` 本体、fvcore FLOPs 工具。
`SegMANDecoder` 的 `forward_mlp_decoder` / `forward_winssm` / `__init__` 模块表迁移到了
`rrs/mmscope_pixel_decoder.py`（按原语句顺序改写、可逐行比对，**未做数值对照**，
见该文件头的迁移表与 `docs/MMSCopE与评分头接入说明_20261016.md` 第 8.1 节）；`cls_seg`（ADE20K 类别 logits）
按要求不迁移。

## 真实算子要求

`segman/mmscope_core.py::require_scan_ops()` 在构造 `VSSM` 时检查
`selective_scan_cuda_oflex` 与 `triton`，缺失即抛 `ImportError` 并给出安装命令。
**没有任何替代实现**：不用卷积、池化、注意力或 Identity 冒充 selective scan。
`use_mmscope=False` 时 `rrs/model.py` 不导入本目录任何模块，旧 B0/HED 在没有这些依赖的
环境里照常运行。安装步骤见 `../../scripts/setup_mmscope.sh`。
