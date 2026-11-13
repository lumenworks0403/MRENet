"""模型配置与同架构恢复。SPED/MCPE/RCA 的旧接口名继续可用。"""

DN_DEFAULTS = dict(
    label_book_size=1,
    label_noise_scale=0.5,
    box_noise_scale=1.0,
    num_dn_queries=100,
    reinit_rate=0.5,
    loss_coef=1.0,
    branch_ref_mode="sigmoid",
)

MM_DEFAULTS = dict(
    channels=128,
    feat_proj_dim=256,
    norm="gn",
    gn_groups=32,
    scan_fp32=True,
    drop_path=0.0,
)
MM_NORMS = ("gn", "bn")

MODE_FLAGS = (("use_hed", False), ("hed_mask_mode", "avg"), ("use_dn", False),
              ("use_mmscope", False), ("use_ref_score", False))


def resolve_model_mode(cfg, checkpoint=None, opts=None):
    """命令行覆盖优先于 checkpoint 模式，checkpoint 优先于新训练配置。"""
    aliases = {
        "use_sped": "use_hed",
        "use_mcpe": "use_mmscope",
        "use_rca": "use_ref_score",
        "mcpe": "mm",
        "sped_mask_mode": "hed_mask_mode"
    }

    def normalize(data):
        out = dict(data)
        for new, old in aliases.items():
            if new in out:
                out[old] = out.pop(new)
        return out

    cfg = normalize(cfg)
    if checkpoint is not None:
        checkpoint = {**checkpoint, "cfg": normalize(checkpoint.get("cfg", {}))}
    mapped_opts = []
    for option in opts or []:
        key, sep, value = option.partition("=")
        head, dot, tail = key.partition(".")
        mapped_opts.append(aliases.get(head, head) + (dot + tail if dot else "") + sep + value)
    # load_cfg 已将命令行值写入原配置。先标准化再重放，防止旧字段覆盖
    # 被同一配置中的论文术语别名遮蔽（例如 use_sped=True + --opts use_hed=False）。
    import ast
    for option in mapped_opts:
        key, separator, value = option.partition("=")
        if not separator:
            raise ValueError(f"配置覆盖必须使用 key=value: {option}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    opts = mapped_opts
    resolved = dict(cfg)
    explicit = {item.split("=", 1)[0] for item in (opts or [])}
    saved = checkpoint.get("cfg", {}) if checkpoint is not None else None
    for key, default in MODE_FLAGS:
        if saved is not None and key not in explicit:
            resolved[key] = saved.get(key, default)
        else:
            resolved[key] = cfg.get(key, default)

    resolved["dn"] = _resolve_sub(cfg, saved, explicit, "dn", DN_DEFAULTS)
    resolved["mm"] = _resolve_sub(cfg, saved, explicit, "mm", MM_DEFAULTS)
    dn, mm = resolved["dn"], resolved["mm"]

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
        raise ValueError("dn.reinit_rate (分支再生概率 p) 必须在 [0, 1]")
    if dn["branch_ref_mode"] not in ("sigmoid", ):
        raise ValueError("dn.branch_ref_mode 必须是 'sigmoid'")
    if mm["norm"] not in MM_NORMS:
        raise ValueError(f"mm.norm 必须是 {MM_NORMS} 之一")
    for key in ("channels", "feat_proj_dim", "gn_groups"):
        if type(mm[key]) is not int or mm[key] <= 0:
            raise ValueError(f"mm.{key} 必须是正整数")
    if mm["norm"] == "gn" and (mm["channels"] % mm["gn_groups"]
                               or mm["feat_proj_dim"] % mm["gn_groups"]):
        raise ValueError("mm.norm='gn' 要求 mm.channels 与 mm.feat_proj_dim 都能被 mm.gn_groups 整除")
    if mm["norm"] == "gn" and min(mm["channels"], mm["feat_proj_dim"]) // mm["gn_groups"] < 2:
        raise ValueError("全局池化分支要求 GroupNorm 每组至少两个通道")
    if type(mm["scan_fp32"]) is not bool:
        raise ValueError("mm.scan_fp32 必须是 True 或 False（布尔值）")
    if not 0.0 <= float(mm["drop_path"]) < 1.0:
        raise ValueError("mm.drop_path 必须在 [0, 1)")
    if resolved["ref_score_loss_coef"] < 0:
        raise ValueError("ref_score_loss_coef 不能为负")
    return resolved


def _resolve_sub(cfg, saved, explicit, name, defaults):
    """嵌套配置优先级：默认值 < checkpoint/新配置 < 命令行子项。"""
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
    name = "SPED+DN" if cfg.get("use_dn", False) else "SPED/no-DN"
    extra = [
        n for n, on in (("MCPE", cfg.get("use_mmscope", False)), ("RCA",
                                                                  cfg.get("use_ref_score", False)))
        if on
    ]
    return "+".join([name] + extra)
