# MRENet

面向低空无人机指代分割，按照所提供论文的方法第 3 节组织为 MCPE、SPED 和 RCA。
输入图像和自然语言表达，输出单个指代目标的二值掩码。

## 方法

- **MCPE**：图文融合后的 1/8、1/16、1/32 特征独立投影并对齐；三路空间重编码后进行二维选择性扫描；将上下文重新注入三组 stage 特征，再与 1/4 浅层特征融合。
- **SPED**：保留六个预训练解码层，重连为一层共享加五个独立并行分支。分支读取共享查询，参考框在送入分支时停止梯度。保留共享层和五个分支，共六组候选。
- **DN**：仅训练时生成正负目标重建查询；普通查询不能读取 DN，DN 组间隔离。每个并行分支以 0.5 概率重新生成整批 DN 输入；概率为 0 时仍计算 DN 损失。
- **RCA**：六组共享 `Linear(D,D) → ReLU → Linear(D,1)` 评分头，各组独立选取最高分查询。每张选中掩码先恢复原图尺寸，再 sigmoid，最后对六张概率掩码等权平均并按 `> 0.5` 阈值化。

源代码按论文实现；PDF 未唯一指定的扫描块内部参数化记录在 [实现说明](docs/IMPLEMENTATION.md)。不附带训练数据或训练后权重。

## 安装

建议 Python 3.12；先安装与 CUDA 匹配的 PyTorch/torchvision，再安装其余依赖：

```bash
pip install -r requirements.txt
bash scripts/download_weights.sh
```

预训练权重路径为 `weights/groundingdino_swint_ogc.pth` 和 `weights/bert-base-uncased/`。
MCPE 使用项目内的可微 PyTorch 扫描，支持 CPU 和 CUDA，不需要额外编译扫描扩展。
GroundingDINO 的 deformable attention 未编译时会使用其 PyTorch 后备实现；CUDA 环境可运行 `bash scripts/build_groundingdino.sh` 编译以改善速度。

## 数据与训练

将 RIS-LAD 按原始目录结构放入 `data/RIS-LAD`，转换时保留官方划分：

```bash
python tools/prepare_rislad.py --root data/RIS-LAD
python tools/train.py --config configs/mrenet_rislad.py
```

完整配置对应论文设置：短边 800、长边不超过 1333、禁止随机水平翻转、冻结 BERT、batch size 4、无梯度累积、50 epochs、第 30 epoch 后学习率乘 0.1、AdamW。
骨干/其余模块初始学习率分别为 `1e-5/1e-4`，权重衰减 `1e-4`，启用 AMP 和 DN。
资源受限时可显式覆盖 batch size，但这会偏离原实验设置。

```bash
python tools/train.py --config configs/mrenet_rislad.py --opts batch_size=1
python tools/train.py --config configs/mrenet_rislad.py --resume work_dirs/mrenet_rislad/last.pth
python tools/train.py --config configs/mrenet_rislad.py --eval-only --resume work_dirs/mrenet_rislad/best.pth
python tools/infer.py --ckpt work_dirs/mrenet_rislad/best.pth --image example.jpg --text "the white car on the left" --out prediction.jpg
```

推理输出叠加图和对应的 `_mask.png`。`queries`/`scores` 各含六项，代表每组选中的候选；它们共同生成一个目标掩码。

## 配置与消融

主配置采用 `use_sped/use_mcpe/use_rca`、`mcpe` 和 `sped_mask_mode`。
内部保留 `use_hed/use_mmscope/use_ref_score`、`mm`、`hed_mask_mode` 作为兼容接口，分别映射到上述模块。
两套字段均可用 `--opts` 显式覆盖；同一条命令建议只使用一套命名。
原 `b0_*` 配置作为基线和历史消融保留，其训练轮数、batch size、DN 开关不自动等同完整论文配置。

```bash
# 关闭 RCA，采用每组有效 token 的最大匹配概率选择候选
python tools/train.py --config configs/mrenet_rislad.py --opts use_rca=False
# 保留 DN 监督，所有分支继承共享 DN
python tools/train.py --config configs/mrenet_rislad.py --opts dn.reinit_rate=0
```

RCA 必须与 SPED 同时开启。论文路径是单目标 top1；基线额外保留多目标能力。

## 两阶段小样本迁移

按 PDF 第 4.5.1 节，分别从 RIS-LAD 验证集选出的权重适配 RefSegRS 和 RRSIS-D。
先根据数据转换说明生成各目标域已清理的 `b0/train.jsonl`、`val.jsonl`、`test.jsonl`，然后固定抽样：

```bash
python tools/prepare_fewshot.py --root data/RefSegRS
python tools/prepare_fewshot.py --root data/RRSIS-D
python tools/finetune.py --config configs/mrenet_refsegrs.py --source-ckpt work_dirs/mrenet_rislad/best.pth --shots 100 --seed 0 --eval-test
python tools/finetune.py --config configs/mrenet_rrsisd.py --source-ckpt work_dirs/mrenet_rislad/best.pth --shots 100 --seed 0 --eval-test
```

预算以完整指代样本计数；每张原始图像只取一个表达。每个 seed 的 20/100/200 训练样本集嵌套，共用额外的 20 个验证样本；种子固定为 0、1、2，测试图像与训练/验证隔离。
每个目标域、每个预算及每个种子都应从同一 RIS-LAD 源模型独立开始；不是从另一个目标域或较小预算继续训练。

阶段1冻结骨干和 BERT，任务模块学习率为 `1e-4`；阶段2解冻骨干，其学习率为 `1e-6`，任务模块为 `1e-5`，BERT 始终冻结。
batch size 2、无梯度累积、启用 AMP 和 DN、禁止翻转。每 50 次成功更新验证一次；连续3次未提升至少0.1个百分点时切换阶段，阶段1最多300次更新。
阶段2每3次停滞将学习率乘0.1，连续6次停滞早停；两阶段总预算1000次更新。阶段切换重置停滞计数，最终按小样本验证 mIoU 保存 `best.pth`。
`--eval-test` 只在训练完成后对选出的最佳权重测试一次，不参与模型选择。

目标域配置专用于 `tools/finetune.py`，普通 `tools/train.py` 会拒绝这些配置，避免误用完整目标域数据替代小样本协议。

## 检查

```bash
python -m unittest discover -s tests -v
```

包括真实选择性扫描与逐时间步递推的输出/梯度对照、非方形特征的前后向、逐组不同查询的选择与融合、DN 隔离、各层匹配监督和权重加载规则。
可以在不下载权重和数据的情况下运行缩小模型的合成检查：

```bash
python tools/smoke_synthetic.py
```

需要真实数据和完整预训练权重的训练冒烟检查为：

```bash
python tools/smoke_mm_score.py --config configs/mrenet_rislad.py
```

历史 MCPE 原型权重的扫描结构与本版不同，不能保证直接恢复。建议从 GroundingDINO 预训练权重重新训练，或显式使用 `--init-from` 从 FPN 基线初始化。
本版修改后的选择规则也会改变旧模型的推理结果；测试通过不代表复现了论文精度或运行速度。

第三方基础组件及许可证见 [third_party/README.md](third_party/README.md)。
