"""模型模式配置；不包含模型参数或训练恢复逻辑。"""

# HED+DN 的默认值与出处 (官方 FT-FSOD 根项目, commit c68cea00):
#   label_noise_scale / box_noise_scale / num_dn_queries:
#       configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py 的 dn_cfg
#   reinit_rate (论文 τ, Eq.7): final_configs/..._ArTaxOr_10shot.py 的 rand_dnquery_rate=0.5,
#       论文表 5/表 6 同为 0.5
#   label_book_size: 官方 = bbox_head.num_classes (数据集类别数)。B0 文本指代按
#       "一条描述 = 类别 0" 处理 (方案 A), 故为 1。属于本项目适配。
#   loss_coef (论文 λ_dn, Eq.4): 官方代码无额外系数, 等价于 1.0。
#   branch_ref_mode: "sigmoid" 为本项目选择, "official_raw" 逐字照搬官方 (见 hed_decoder.py)。
DN_DEFAULTS = dict(
    label_book_size=1,
    label_noise_scale=0.5,
    box_noise_scale=1.0,
    num_dn_queries=100,
    reinit_rate=0.5,
    loss_coef=1.0,
    branch_ref_mode="sigmoid",
)

# MMSCopE 像素支路 (use_mmscope) 的内部配置。默认值依据见
# docs/MMSCopE与评分头接入说明_20261016.md 与 rrs/mmscope_pixel_decoder.py 文件头:
#   channels=128       官方 SegMAN-T 解码器宽度 (segman_t_ade.py channels=128); 扫描通道 3*128。
#   feat_proj_dim=256  与 GroundingDINO hidden_dim 相同, 逐层投影不扩张也不压缩。
#   norm="gn"/32       官方用 SyncBN; 本工程 batch=1~2 单卡, 且官方 image_pool 在 1x1 特征上
#                      做 BN 在 batch=1 时直接报错, 故默认 GroupNorm(32) (与现有 FPN 一致)。
#                      "bn" 可切回 BatchNorm 作对照。
#   scan_fp32=True     只在 SS2D 扫描处关 autocast 转 fp32 (算子要求 dtype 一致), 其余仍按 AMP。
#   drop_path=0.0      官方 VSSMBlock 默认值。
# 本文件不导入 torch (tests/test_mode_entrypoints.py 在无 torch 环境下直接 import 它),
# 因此这里只放纯数据; 真实模块在 rrs/mmscope_pixel_decoder.py。
MM_DEFAULTS = dict(
    channels=128,
    feat_proj_dim=256,
    norm="gn",
    gn_groups=32,
    scan_fp32=True,
    drop_path=0.0,
)
MM_NORMS = ("gn", "bn")

# 新增的模式字段与各自默认值 (旧 checkpoint / 旧配置缺字段时按这里的默认处理)。
MODE_FLAGS = (("use_hed", False), ("hed_mask_mode", "avg"), ("use_dn", False),
              ("use_mmscope", False), ("use_ref_score", False))


def resolve_model_mode(cfg, checkpoint=None, opts=None):
    """返回新配置：命令行显式覆盖 > checkpoint cfg > 新训练配置。

    老 checkpoint 缺少 use_hed / use_dn / use_mmscope / use_ref_score 时按 False 恢复，
    不能从文件名猜模式。只恢复 MODE_FLAGS 列出的模式字段与 dn / mm 配置块，
    数据路径、学习率、ref_score_loss_coef 等仍使用调用方配置。
    命令行可覆盖嵌套子项 (如 --opts dn.reinit_rate=0 mm.channels=96)。

    注意 checkpoint 参数只用于 **同架构恢复** (--resume / eval-only / infer)。
    --init-from 是权重初始化, 结构由新配置决定, 调用方必须传 checkpoint=None
    (见 tools/train.py)，否则旧 checkpoint 缺少的新标志会把 MM/Score 意外关掉。
    """
    resolved = dict(cfg)
    explicit = {item.split("=", 1)[0] for item in (opts or [])}
    saved = checkpoint.get("cfg", {}) if checkpoint is not None else None
    for key, default in MODE_FLAGS:
        if saved is not None and key not in explicit:
            resolved[key] = saved.get(key, default)
        else:
            resolved[key] = cfg.get(key, default)

    # dn / mm 子项: 默认值 < (新配置 或 checkpoint) < 命令行显式子项
    resolved["dn"] = _resolve_sub(cfg, saved, explicit, "dn", DN_DEFAULTS)
    resolved["mm"] = _resolve_sub(cfg, saved, explicit, "mm", MM_DEFAULTS)
    dn, mm = resolved["dn"], resolved["mm"]

    # 评分损失权重: 与其它训练超参一样由当前配置决定, 不从 checkpoint 恢复 (它不是结构标志)。
    resolved["ref_score_loss_coef"] = float(cfg.get("ref_score_loss_coef", 1.0))

    for key in ("use_hed", "use_dn", "use_mmscope", "use_ref_score"):
        if type(resolved[key]) is not bool:
            raise ValueError(f"{key} 必须是 True 或 False（布尔值）")
    if resolved["hed_mask_mode"] not in ("avg", "last"):
        raise ValueError("hed_mask_mode 必须是 'avg' 或 'last'")
    if resolved["use_dn"] and not resolved["use_hed"]:
        raise ValueError("本阶段不支持 B0+DN (use_hed=False, use_dn=True)")
    if resolved["use_ref_score"] and not resolved["use_hed"]:
        raise ValueError("本阶段 use_ref_score=True 只支持 use_hed=True")
    if not 0.0 <= float(dn["reinit_rate"]) <= 1.0:
        raise ValueError("dn.reinit_rate (论文 τ) 必须在 [0, 1]")
    if dn["branch_ref_mode"] not in ("sigmoid", "official_raw"):
        raise ValueError("dn.branch_ref_mode 必须是 'sigmoid' 或 'official_raw'")
    if mm["norm"] not in MM_NORMS:
        raise ValueError(f"mm.norm 必须是 {MM_NORMS} 之一")
    for key in ("channels", "feat_proj_dim", "gn_groups"):
        if type(mm[key]) is not int or mm[key] <= 0:
            raise ValueError(f"mm.{key} 必须是正整数")
    if mm["norm"] == "gn" and (mm["channels"] % mm["gn_groups"]
                               or mm["feat_proj_dim"] % mm["gn_groups"]):
        raise ValueError("mm.norm='gn' 要求 mm.channels 与 mm.feat_proj_dim 都能被 mm.gn_groups 整除")
    if type(mm["scan_fp32"]) is not bool:
        raise ValueError("mm.scan_fp32 必须是 True 或 False（布尔值）")
    if not 0.0 <= float(mm["drop_path"]) < 1.0:
        raise ValueError("mm.drop_path 必须在 [0, 1)")
    if resolved["ref_score_loss_coef"] < 0:
        raise ValueError("ref_score_loss_coef 不能为负")
    return resolved


def _resolve_sub(cfg, saved, explicit, name, defaults):
    """嵌套配置块 (dn / mm) 的优先级: defaults < (checkpoint 或 新配置) < 命令行显式子项。"""
    base = saved.get(name, {}) if saved is not None else cfg.get(name, {})
    out = {**defaults, **(base or {})}
    for k in defaults:
        if f"{name}.{k}" in explicit:
            out[k] = cfg[name][k]
    unknown = set(out) - set(defaults)
    if unknown:
        raise ValueError(f"未知的 {name} 配置项: {sorted(unknown)}")
    return out


def model_variant(cfg):
    if not cfg["use_hed"]:
        return "B0"
    name = "HED+DN" if cfg.get("use_dn", False) else "HED-structure/no-DN"
    extra = [
        n for n, on in (("MM", cfg.get("use_mmscope", False)), ("Score",
                                                                cfg.get("use_ref_score", False)))
        if on
    ]
    return "+".join([name] + extra)
