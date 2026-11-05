# 第三方来源

| 组件 | 来源 / 版本 | 本仓库保留内容 |
|---|---|---|
| GroundingDINO | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，原项目记录 commit `856dde20aee659246248e20734ef9ba5214f5e44` | `groundingdino/` 源码、CUDA 内核、`setup.py`、依赖与许可证 |
| SegMAN / MMSCopE | [yunxiangfu2001/SegMAN](https://github.com/yunxiangfu2001/SegMAN)，commit `9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8` | 原项目抽取的 MMSCopE 模块、对照源码与 selective scan 内核；详情见 [VERSION.md](SegMAN/VERSION.md) |
| Open-GroundingDino | 原项目记录 commit `d248268`，训练超参与匹配/损失参考来源 | 保留 [许可证](licenses/Open-GroundingDino.LICENSE)；完整训练仓库未被运行代码导入，因此从发布版移除 |
| FT-FSOD | 原项目记录 commit `c68cea00fac282ec526d59bca55f27b3203daf37` | HED/DN 在 `rrs/` 中的迁移说明、来源文件与 SHA256 记录；未附上游仓库 |

本次整理使用原项目已有源码，没有联网重新核验上述 commit。GroundingDINO 与 SegMAN 保留文件的内容与整理前一致。GroundingDINO 官方示例、图片、Notebook、Docker 配置和独立 Open-GroundingDino 仓库未打入发布包。

`scripts/build_groundingdino.sh` 仅在 `work_dirs/` 临时副本中适配 PyTorch 的 `scalar_type()`、`is_cuda()`、`data_ptr()` 接口，编译出的扩展复制回源码目录并由 `.gitignore` 排除。`setup_mmscope.sh` 同样使用临时源码构建。

GroundingDINO 与 SegMAN 目录附有上游 LICENSE；SegMAN 部分文件还保留 NVIDIA 许可声明。请以对应文件的具体声明为准。本次整理没有替换或重新授权第三方代码。
