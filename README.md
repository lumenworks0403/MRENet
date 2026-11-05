<div align="center">

# MRENet

### Low-Altitude UAV Referring Image Segmentation<br>via Multiscale Contextual Pixel Enhancement<br>and Referentially Guided Ensemble Prediction

**Shuo Zhang · Quanzhen Chen · Li Yu · Ming Li**<br>
**Haoyu Pan · Yilin Li · Zhiyi Shi · Edmond Q. Wu**

**Multiscale context. Parallel decoding. Referential candidate selection.**

![Task](assets/badges/task.svg)
![Domain](assets/badges/domain.svg)
![Framework](assets/badges/framework.svg)
![Encoders](assets/badges/encoders.svg)

**[Prepare Data](#data-preparation) &nbsp;·&nbsp; [Getting Started](#getting-started)**

[Dataset](#dataset) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Method](#method) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Getting Started](#getting-started) &nbsp;·&nbsp; [Code Guide](#code-guide)

</div>

---

## Dataset

**RIS-LAD** is the primary benchmark for low-altitude UAV referring image segmentation in this study. It covers eight object categories, including pedestrians, vehicles, and vessels. Each sample pairs an image with a natural-language expression and the mask of the referred target. Small objects, dense groups of similar instances, and complex backgrounds make precise segmentation challenging.

The manuscript uses the original RIS-LAD splits for training, model selection, and testing, and evaluates few-shot transfer on **RefSegRS** and **RRSIS-D**. Counts below are referring samples from Table 1.

| Dataset | Train | Validation | Test | Role |
| :--- | ---: | ---: | ---: | :--- |
| **RIS-LAD** | 9,712 | 1,396 | 2,763 | Source training and primary evaluation |
| **RefSegRS** | 2,172 | 431 | 1,817 | Few-shot adaptation and transfer evaluation |
| **RRSIS-D** | 12,181 | 1,740 | 3,481 | Few-shot adaptation and transfer evaluation |

Target-domain training counts describe the sampling pools. The main transfer experiments use **100 training annotations + 20 validation annotations per target domain**, with results averaged over three random seeds.

<details>
<summary><b>Data availability and valid samples</b></summary>

Obtain each dataset from its original provider, then follow [Prepare Data](#data-preparation). This repository does not redistribute dataset images or annotations.

The conversion described in the manuscript retains all RefSegRS samples. After removing RRSIS-D records with empty masks or inconsistent image/mask dimensions, the effective train, validation, and test counts are **12,170 / 1,738 / 3,477**. Record the effective sample counts when reporting results.

Dataset layouts and conversion details are documented in the [data conversion notes](docs/数据集格式转换说明_20261025.md).

</details>

## Overview

MRENet segments the target specified by a natural-language expression in low-altitude UAV imagery. It addresses two related challenges: **limited spatial detail in small targets** and **ambiguity among visually similar instances**.

Built on GroundingDINO, the framework uses **MCPE** to enhance pixel representations, **SPED** to produce multiple query sets, and **RCA** to assess referential relevance before mask aggregation.

<table align="center">
  <tr>
    <th align="center">48.85%</th>
    <th align="center">52.41%</th>
    <th align="center">+6.17 pp</th>
  </tr>
  <tr>
    <td align="center">RIS-LAD Test mIoU</td>
    <td align="center">RIS-LAD Test oIoU</td>
    <td align="center">mIoU gain over B0</td>
  </tr>
</table>

<p align="center"><sub>Results reported in Tables 2 and 4 of the manuscript; not newly reproduced measurements from this code release.</sub></p>

## Method

<p align="center">
  <a href="assets/figures/figure1-overview.png">
    <img src="assets/figures/figure1-overview.png" alt="Figure 1: MRENet architecture, including visual and text encoding, MCPE, SPED, RCA, and mask aggregation" width="100%">
  </a>
</p>

<p align="center"><sub><b>Figure 1. Overall framework.</b> Click the image for the full-resolution version, or open the <a href="assets/figures/figure1-overview.pdf">vector PDF</a> to zoom in.</sub></p>

| Component | Role |
| :--- | :--- |
| **GroundingDINO · Swin-T + BERT** | Extract visual and text features and initialize cross-modal queries for target localization. |
| **Multiscale Contextual Pixel Enhancement (MCPE)** | Combine multiscale context re-encoding with shallow-detail fusion to enhance pixel features for mask prediction. |
| **Shared Parallel Ensemble Decoder (SPED)** | Use one shared layer and five independently parameterized parallel layers to produce six query sets. |
| **Referential Candidate Assessment (RCA)** | Score candidate queries for referential relevance to support target selection. |
| **Auxiliary denoising (DN)** | Supervise perturbed queries, with stochastic regeneration of branch-specific denoising inputs. |
| **Probability-mask aggregation** | Restore selected masks to the original image size and combine their probabilities before binarization. |

The implementation retains the historical **B0_HED** layout. `MMSCopE`, `HED`, and `RefScoreHead` identify the pixel-enhancement, decoder, and candidate-scoring implementations. See the [Code Guide](#code-guide).

<details>
<summary><b>Implementation and reproduction status</b></summary>

This release organizes the existing implementation while preserving its behavior. Full manuscript alignment and independent benchmark reproduction remain pending.

- **Candidate selection:** the manuscript selects a candidate independently in each output layer. The current `rrs/evaluate.py` averages candidate scores across six layers and selects one shared query index. This difference needs to be reconciled with the original experiment records.
- **Training recipe:** the manuscript uses 50 epochs, batch size 4, no gradient accumulation, and DN. The existing combined configuration defaults to 12 epochs, batch size 2 with two accumulation steps, and DN disabled. Commands below run the current configurations.
- **Few-shot transfer:** dataset readers and standard training configurations are available; the complete three-seed, nested-budget, two-stage adaptation protocol is not yet included.
- **Validation:** unit tests check implementation behavior. Full training, real SS2D CUDA execution, and benchmark results require separate GPU validation. Trained task checkpoints are not included.

</details>

## Results

### RIS-LAD benchmark

**Test results from Table 2 of the manuscript.** Values are percentages; higher is better.

| Method | mIoU ↑ | oIoU ↑ |
| :--- | ---: | ---: |
| LAVT | 30.14 | 41.97 |
| ASDA | 33.33 | 37.53 |
| RMSIN | 39.60 | 48.82 |
| VATEX | 18.53 | 24.27 |
| LGCE | 26.17 | 40.75 |
| FIANet | 37.44 | 43.39 |
| RSRefSeg | 41.16 | 47.71 |
| CADFormer | 39.32 | 46.47 |
| SAARN | 41.67 | 49.60 |
| **MRENet (Ours)** | **48.85** | **52.41** |

<details>
<summary><b>Precision at different IoU thresholds</b></summary>

Prec@X for the full model, from Table 2. It measures the percentage of samples with mask `IoU >= X`; these IoU cutoffs differ from the probability threshold used for binarization.

| Prec@0.5 ↑ | Prec@0.6 ↑ | Prec@0.7 ↑ | Prec@0.8 ↑ | Prec@0.9 ↑ |
| ---: | ---: | ---: | ---: | ---: |
| 55.01 | 52.73 | 48.35 | 41.77 | 27.80 |

</details>

<details>
<summary><b>Module and denoising ablations</b></summary>

RIS-LAD test results from Table 4. B0 is the internal GroundingDINO-based baseline.

| Configuration | MCPE | SPED | RCA | Denoising | mIoU ↑ | oIoU ↑ |
| :--- | :---: | :---: | :---: | :--- | ---: | ---: |
| A · B0 | — | — | — | Disabled | 42.68 | 48.37 |
| B | ✓ | — | — | Disabled | 44.62 | 49.53 |
| C | — | ✓ | — | Disabled | 44.03 | 49.28 |
| D | ✓ | ✓ | — | Disabled | 45.86 | 50.26 |
| E | ✓ | ✓ | ✓ | Disabled | 47.58 | 51.54 |
| F | ✓ | ✓ | ✓ | Enabled, p = 0 | 48.32 | 52.01 |
| **G · Full model** | **✓** | **✓** | **✓** | **Enabled, p = 0.5** | **48.85** | 52.41 |
| H | ✓ | ✓ | ✓ | Enabled, p = 1 | 48.53 | **52.46** |

`p` is the probability of regenerating the entire DN input set for a parallel branch. DN supervision remains active at `p = 0`. The full model improves mIoU and oIoU over B0 by **6.17** and **4.04 percentage points**, respectively.

</details>

<details>
<summary><b>Few-shot transfer to remote sensing datasets</b></summary>

Manuscript results with 100 training and 20 separate validation annotations per target domain, averaged over three seeds.

| Target domain | Train / validation annotations | mIoU ↑ | oIoU ↑ |
| :--- | :---: | ---: | ---: |
| RefSegRS | 100 / 20 | 39.12 | 54.93 |
| RRSIS-D | 100 / 20 | 38.46 | 49.08 |

The two target domains are adapted independently from the RIS-LAD source model. Test sets are excluded from sampling, model selection, and threshold tuning. See the implementation-status note above for current protocol support.

</details>

## Getting Started

### 1. Installation

Run commands from the repository root. The full pixel-enhancement module requires **Linux, an NVIDIA GPU, Triton, and the real selective scan CUDA operator**.

```bash
conda create -n mrenet python=3.10 -y
conda activate mrenet
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
bash scripts/build_groundingdino.sh
bash scripts/download_weights.sh
```

When enabling `use_mmscope=True`, install and check SS2D:

```bash
bash scripts/setup_mmscope.sh
python tools/check_ss2d.py
```

<details>
<summary><b>Environment versions, server setup, and pretrained weights</b></summary>

The commands above follow the current installation example; not every CUDA/GPU combination has been verified. The manuscript reports Python 3.12.3, PyTorch 2.8.0, CUDA 12.8, cuDNN 9.10.2, and an RTX 3080 Ti. Record the actual environment used for an experiment.

Alternative setup scripts:

```bash
bash scripts/setup_server.sh
bash scripts/setup_server.sh --existing-env
bash scripts/setup_autodl.sh
```

Use `HF_ENDPOINT`, `CUDA_HOME`, and `TORCH_CUDA_ARCH_LIST` to configure downloads and compilation. The default PyTorch 2.4.1 setup supports `cu118`, `cu121`, and `cu124`. Compiling CUDA extensions requires a compatible CUDA toolkit and `nvcc`.

Default pretrained-resource layout:

```text
weights/
├── groundingdino_swint_ogc.pth
└── bert-base-uncased/
    ├── config.json
    ├── vocab.txt
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── model.safetensors
```

Official GroundingDINO weights provide initialization, not a trained referring-segmentation checkpoint. Evaluation and prediction below require a checkpoint produced by the corresponding training run.

Without the GroundingDINO `_C` extension, B0/HED can use the existing PyTorch attention path at reduced speed. The complete MMSCopE module still requires the real SS2D CUDA implementation; CPU tests do not validate it.

</details>

<a name="data-preparation"></a>

### 2. Prepare data

Place the original datasets under `data/`, then generate unified JSONL annotations:

```bash
python tools/prepare_rislad.py --root data/RIS-LAD
python tools/prepare_remote_sensing.py --dataset refsegrs --data-root data
python tools/prepare_remote_sensing.py --dataset rrsisd --data-root data
```

RIS-LAD layout:

```text
data/RIS-LAD/
├── ris_lad/
│   ├── instances.json
│   └── refs(unc).p
├── images/ris_lad/
└── b0/                         # Generated annotations
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

See the [data conversion notes](docs/数据集格式转换说明_20261025.md) for RefSegRS and RRSIS-D layouts.

<details>
<summary><b>Annotation format and other datasets</b></summary>

Each JSONL row represents one referring expression. `image` is relative to `image_root`; `bbox` uses COCO `xywh` coordinates in pixels; `segmentation` accepts COCO polygons or RLE.

```json
{"image":"images/example.jpg","caption":"the car on the left","height":512,"width":512,"id":"sample_1","objects":[{"bbox":[10,20,30,40],"segmentation":[[10,20,40,20,40,60,10,60]]}]}
```

`objects` may be empty or contain multiple targets. RefCOCO supports the `jxu124/refcoco` parquet annotations. Run `python tools/convert_refdrone.py --help` for RefDrone options. Box-filled masks are for pipeline checks only, not substitutes for segmentation annotations.

</details>

### 3. Train

Train the existing HED configuration:

```bash
python tools/train.py --config configs/b0_hed_rislad.py
```

Train HED + MMSCopE + the scoring head, explicitly enabling DN:

```bash
python tools/train.py --config configs/b0_hed_mm_score_rislad.py --opts use_dn=True
```

These commands use the current recipes described in the implementation-status note. Logs and checkpoints are written to `output_dir`; model selection uses validation mIoU.

<details>
<summary><b>Model configurations</b></summary>

| Mode | RIS-LAD configuration | Notes |
| :--- | :--- | :--- |
| B0 | [b0_rislad.py](configs/b0_rislad.py) | Baseline |
| HED | [b0_hed_rislad.py](configs/b0_hed_rislad.py) | Shared parallel decoder |
| HED + DN | [b0_hed_dn_rislad.py](configs/b0_hed_dn_rislad.py) | DN is used only during training |
| HED + MMSCopE | [b0_hed_mm_rislad.py](configs/b0_hed_mm_rislad.py) | Requires real SS2D |
| HED + Score | [b0_hed_score_rislad.py](configs/b0_hed_score_rislad.py) | Referential scoring head |
| HED + MMSCopE + Score | [b0_hed_mm_score_rislad.py](configs/b0_hed_mm_score_rislad.py) | DN disabled by default; enable explicitly |

Configurations use `_base_ = "base.py"` inheritance and are loaded by `rrs.config.load_cfg()`. Dictionary fields are replaced as a whole; use `dict(train_data, max_samples=16)` or dotted command-line overrides for local changes.

Current RIS-LAD defaults are `batch_size=1, grad_accum=1` for B0 and `batch_size=2, grad_accum=2` for HED. Match actual training settings in controlled comparisons. Equal effective batch sizes do not guarantee identical padding or numerical behavior.

</details>

<details>
<summary><b>Resume training and initialize a different architecture</b></summary>

```bash
# Restore the same architecture, optimizer, scheduler, and epoch
python tools/train.py --config configs/b0_hed_rislad.py --resume work_dirs/b0_hed_rislad_noDN/last.pth --opts gdino_ckpt=None

# Load compatible parameters; initialize newly added modules
python tools/train.py --config configs/b0_hed_mm_score_rislad.py --init-from work_dirs/b0_hed_rislad_noDN/best.pth --opts gdino_ckpt=None
```

`--resume` restores the checkpoint's model mode, with explicit `--opts` taking precedence. `--init-from` uses the current configuration and cannot be combined with `--resume` or `--eval-only`. Current resume behavior does not guarantee bitwise equivalence to uninterrupted training.

</details>

### 4. Evaluate

Evaluate on the validation set:

```bash
python tools/train.py --config configs/b0_hed_rislad.py --eval-only --resume work_dirs/b0_hed_rislad_noDN/best.pth --opts gdino_ckpt=None
```

For final testing, explicitly select the test annotations. Continue to use validation data for model selection during training:

```bash
python tools/train.py --config configs/b0_hed_rislad.py --eval-only --resume work_dirs/b0_hed_rislad_noDN/best.pth --opts gdino_ckpt=None val_data.path=data/RIS-LAD/b0/test.jsonl output_dir=work_dirs/test_rislad
```

Metrics include mIoU, oIoU, and Prec@0.5/0.6/0.7/0.8/0.9. Masks are restored to the original image size and binarized with probability `> 0.5`; Prec@X uses `IoU >= X`.

### 5. Predict

```bash
python tools/infer.py --ckpt work_dirs/b0_hed_rislad_noDN/best.pth --image example.jpg --text "the white car on the left" --out work_dirs/prediction.jpg
```

The command saves an overlay and a corresponding `_mask.png` binary mask. Model settings are restored from the checkpoint. Use `--hed-mask-mode avg` or `last` to select mask aggregation. The default resize targets a shorter side of 800 pixels and a maximum longer side of 1333; override with `--short-side` and `--max-size`.

## Code Guide

<details>
<summary><b>Repository structure</b></summary>

```text
MRENet/
├── configs/        Dataset, model, and diagnostic configurations
├── rrs/            Models, training utilities, data, losses, and postprocessing
├── tools/          Training, evaluation, prediction, conversion, and checks
├── scripts/        Environment setup, downloads, and CUDA builds
├── tests/          CPU unit tests
├── docs/           Implementation and data-conversion notes
├── assets/         Framework figure, vector PDF, and badges
├── third_party/    Required upstream sources, versions, and licenses
├── .github/        Automated checks
├── requirements.txt
└── README.md
```

</details>

| Component | Implementation |
| :--- | :--- |
| Main architecture and checkpoint loading | [model.py](rrs/model.py) |
| Pixel enhancement corresponding to MCPE | [mmscope_pixel_decoder.py](rrs/mmscope_pixel_decoder.py) |
| Shared parallel decoder corresponding to SPED | [hed_decoder.py](rrs/hed_decoder.py) |
| RCA candidate-scoring head | [ref_score_head.py](rrs/ref_score_head.py) |
| Denoising queries and branch sampling | [dn.py](rrs/dn.py) |
| Matching and training losses | [criterion.py](rrs/criterion.py) |
| Candidate selection, mask restoration, and metrics | [evaluate.py](rrs/evaluate.py) |
| Dataset and preprocessing | [data.py](rrs/data.py) · [preprocess.py](rrs/preprocess.py) |
| Configuration and training utilities | [config.py](rrs/config.py) · [training.py](rrs/training.py) |
| Training and evaluation entry point | [train.py](tools/train.py) |
| Single-image prediction | [infer.py](tools/infer.py) |

<details>
<summary><b>Development checks and GPU validation</b></summary>

CPU unit tests and syntax checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q rrs tools configs tests
```

Tests do not download pretrained weights. MMSCopE shape and wiring tests use stubs and do not validate the CUDA scan kernel. The [GitHub Actions workflow](.github/workflows/tests.yml) runs CPU tests and syntax checks.

With the required datasets, weights, and GPU environment:

```bash
python tools/check_b0.py --config configs/b0_refcoco.py
python tools/smoke_dn.py --config configs/b0_hed_dn_rislad.py --iters 2
python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2
bash scripts/verify_mm_score.sh
```

These checks validate the execution pipeline, not complete training or benchmark performance.

</details>

## Citation

Please cite the corresponding manuscript when using this work. Update the publication metadata once the paper is formally published.

```bibtex
@unpublished{zhang2026mrenet,
  title  = {Low-Altitude UAV Referring Image Segmentation via Multiscale
            Contextual Pixel Enhancement and Referentially Guided
            Ensemble Prediction},
  author = {Zhang, Shuo and Chen, Quanzhen and Yu, Li and Li, Ming and
            Pan, Haoyu and Li, Yilin and Shi, Zhiyi and Wu, Edmond Q.},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements and License

This project builds on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and uses or draws on implementations from SegMAN / MMSCopE, Open-GroundingDino, and FT-FSOD. See [third-party sources and licenses](third_party/README.md) for provenance, recorded versions, and retained files.

Adaptation details are recorded in the [HED and DN notes](docs/HED_DN接入说明_20261013.md) and [pixel-decoder and scoring-head notes](docs/MMSCopE与评分头接入说明_20261016.md). Upstream copyright notices and licenses are retained. A top-level license for the original project code has not yet been specified.

<p align="center"><a href="#mrenet">Back to top ↑</a></p>
