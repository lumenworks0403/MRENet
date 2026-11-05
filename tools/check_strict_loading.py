"""严格加载规则的断言检查: 删掉/添加权重键后, load_model_state 与 init_model_weights
必须报错, 且报错信息里要确实提到那些键。不符合预期 -> exit 1 (不是只打印 FAILED)。

    python tools/check_strict_loading.py --ckpt work_dirs/.../last.pth

对"应当报错"的每一项, 同时核对三件事:
  1. 确实抛出了 RuntimeError;
  2. 异常信息里出现了被动过手脚的那个键名 (避免把无关的 RuntimeError 当成通过);
  3. 异常信息落在预期的字段上 (missing= 还是 unexpected=)。

检查项:
  A. 删 mask_head 的键            -> load_model_state 必须报 missing, 且提到该键
  B. 删 ref_score_head.* 的键     -> load_model_state 必须报 missing (评分头不享受豁免)
  C. 删 pixel_decoder.* 的键      -> load_model_state 必须报 missing
  D. 加一个不存在的键             -> load_model_state 必须报 unexpected
  E. 改一个键的形状               -> init_model_weights 必须报形状不匹配
  F. 源/目标都启用评分头但缺键    -> init_model_weights 必须报错 (白名单按源配置收紧后的行为)
  G. 正常同架构加载               -> 必须成功且无缺失键 (对照项, 证明上面不是"总是报错")

本机 (Windows) 未执行本脚本。
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.model import build_model, init_model_weights, load_model_state  # noqa: E402
from rrs.model_mode import resolve_model_mode  # noqa: E402


def expect_raises(fn, must_mention, field, label):
    """fn 必须抛 RuntimeError, 且信息里包含 must_mention 里的每个片段与 field 字段。"""
    try:
        fn()
    except RuntimeError as e:
        msg = str(e)
        missing_bits = [s for s in must_mention if s not in msg]
        field_ok = field is None or field in msg
        return {
            "raised": True,
            "field_ok": field_ok,
            "mentions_all_keys": not missing_bits,
            "not_mentioned": missing_bits,
            "msg": msg[:300],
            "ok": field_ok and not missing_bits
        }
    except Exception as e:  # 非 RuntimeError: 不算"按预期报错"
        return {"raised": True, "ok": False, "unexpected_exception": repr(e)[:300]}
    return {"raised": False, "ok": False, "note": f"{label}: 未报错, 但按规则必须报错"}


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    report, ok = {"ckpt": args.ckpt}, True

    ck = torch.load(args.ckpt, map_location="cpu")
    saved_cfg = ck.get("cfg", {})
    cfg = resolve_model_mode(saved_cfg, ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report["mode"] = {k: cfg[k] for k in ("use_hed", "use_dn", "use_mmscope", "use_ref_score")}

    def fresh():
        m, _ = build_model(cfg["gdino_cfg"], None, cfg["bert_path"], device, use_checkpoint=False,
                           use_hed=cfg["use_hed"], use_dn=cfg["use_dn"], dn_cfg=cfg["dn"],
                           use_mmscope=cfg["use_mmscope"], mm_cfg=cfg["mm"],
                           use_ref_score=cfg["use_ref_score"])
        return m

    model = fresh()
    sd_full = ck["model"]

    def first_key(prefix):
        ks = sorted(k for k in sd_full if k.startswith(prefix))
        return ks[0] if ks else None

    # ---- G) 对照: 正常加载必须成功 ----
    missing = load_model_state(model, dict(sd_full))
    report["G_normal_load"] = {"missing": missing, "ok": not missing}
    ok &= report["G_normal_load"]["ok"]

    # ---- A/B/C) 删键必须报 missing, 且信息里提到该键 ----
    cases = [("A_mask_head", "mask_head."), ("C_pixel_decoder", "pixel_decoder.")]
    if cfg["use_ref_score"]:
        cases.insert(1, ("B_ref_score_head", "ref_score_head."))
    else:
        report["B_ref_score_head"] = {"ok": True, "skipped": "该 checkpoint 未启用评分头"}
    for label, prefix in cases:
        key = first_key(prefix)
        if key is None:
            report[label] = {"ok": False, "note": f"checkpoint 里找不到 {prefix}* 的键"}
            ok = False
            continue
        sd = {k: v for k, v in sd_full.items() if k != key}
        report[label] = expect_raises(lambda sd=sd: load_model_state(fresh(), sd), [key],
                                      "missing=", label)
        report[label]["removed_key"] = key
        ok &= report[label]["ok"]

    # ---- D) 多余键必须报 unexpected ----
    bogus = "ref_score_head.__does_not_exist__"
    sd = dict(sd_full)
    sd[bogus] = torch.zeros(2)
    report["D_unexpected_key"] = expect_raises(lambda: load_model_state(fresh(), sd), [bogus],
                                               "unexpected=", "D")
    ok &= report["D_unexpected_key"]["ok"]

    # ---- E) 形状不符必须在 init_model_weights 里报错 ----
    key = first_key("mask_head.")
    sd = dict(sd_full)
    sd[key] = torch.zeros(tuple(d + 1 for d in sd_full[key].shape))
    report["E_shape_mismatch"] = expect_raises(lambda: init_model_weights(fresh(), sd, saved_cfg),
                                               [key], "形状不匹配", "E")
    ok &= report["E_shape_mismatch"]["ok"]

    # ---- F) 源/目标都启用评分头, 但源缺评分头键 -> init_model_weights 也必须报错 ----
    if cfg["use_ref_score"]:
        key = first_key("ref_score_head.")
        sd = {k: v for k, v in sd_full.items() if k != key}
        # saved_cfg 声称启用了评分头, 所以不能把缺键当成"正常新增"
        report["F_ref_score_incomplete"] = expect_raises(
            lambda: init_model_weights(fresh(), sd, {
                **saved_cfg, "use_ref_score": True
            }), [key], None, "F")
        ok &= report["F_ref_score_incomplete"]["ok"]
    else:
        report["F_ref_score_incomplete"] = {"ok": True, "skipped": "该 checkpoint 未启用评分头"}

    report["PASS"] = bool(ok)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs("work_dirs", exist_ok=True)
    with open("work_dirs/check_strict_loading.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
