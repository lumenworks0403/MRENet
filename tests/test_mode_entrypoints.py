"""无需 torch 的模式/入口回归测试。

直接测试生产 resolve_model_mode；通过 AST 加载生产入口函数原代码，
隔离重型导入，mock 模型构建/IO。不是模型前向或梯度测试。
运行 python tests/test_mode_entrypoints.py。
"""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrs.model_mode import MM_DEFAULTS, MODE_FLAGS, model_variant, resolve_model_mode


def production_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []  # 仅移除 torch.no_grad，函数体保持原样
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class TestMode(unittest.TestCase):

    def test_new_and_legacy_defaults(self):
        self.assertFalse(resolve_model_mode({})["use_hed"])
        self.assertTrue(resolve_model_mode({"use_hed": True})["use_hed"])
        self.assertFalse(resolve_model_mode({"use_hed": True}, {"cfg": {}})["use_hed"])
        self.assertFalse(resolve_model_mode({"use_hed": True}, {})["use_hed"])

    def test_checkpoint_restores_only_mode(self):
        cfg = {"output_dir": "new", "lr": 0.001}
        saved = {"cfg": {"use_hed": True, "hed_mask_mode": "last", "lr": 99}}
        result = resolve_model_mode(cfg, saved)
        self.assertTrue(result["use_hed"])
        self.assertEqual(result["hed_mask_mode"], "last")
        self.assertEqual(result["lr"], cfg["lr"])
        self.assertEqual(result["output_dir"], "new")
        self.assertNotIn("use_hed", cfg)

    def test_explicit_override(self):
        result = resolve_model_mode({
            "use_hed": False,
            "hed_mask_mode": "last"
        }, {"cfg": {
            "use_hed": True,
            "hed_mask_mode": "avg"
        }}, ["use_hed=False", "hed_mask_mode=last"])
        self.assertFalse(result["use_hed"])
        self.assertEqual(result["hed_mask_mode"], "last")

    def test_invalid_values_rejected(self):
        for cfg in ({
                "use_hed": "False"
        }, {
                "use_hed": 1
        }, {
                "hed_mask_mode": "layer2"
        }, {
                "use_hed": True,
                "use_dn": 1
        }, {
                "use_hed": True,
                "use_dn": True,
                "dn": {
                    "reinit_rate": 1.5
                }
        }, {
                "use_hed": True,
                "use_dn": True,
                "dn": {
                    "typo_key": 1
                }
        }):
            with self.assertRaises(ValueError):
                resolve_model_mode(cfg)

    # ---- DN 模式 ----
    def test_dn_default_off_and_legacy(self):
        self.assertFalse(resolve_model_mode({})["use_dn"])
        self.assertFalse(resolve_model_mode({"use_hed": True})["use_dn"])
        # 老 HED-no-DN checkpoint 没有 use_dn -> False
        self.assertFalse(
            resolve_model_mode({
                "use_dn": True,
                "use_hed": True
            }, {"cfg": {
                "use_hed": True
            }})["use_dn"])

    def test_b0_plus_dn_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_mode({"use_hed": False, "use_dn": True})
        with self.assertRaises(ValueError):
            resolve_model_mode({}, {"cfg": {"use_hed": False, "use_dn": True}})

    def test_dn_restored_from_checkpoint_and_override(self):
        saved = {"cfg": {"use_hed": True, "use_dn": True, "dn": {"reinit_rate": 0.25}}}
        r = resolve_model_mode({"use_hed": True, "dn": {"reinit_rate": 0.9}}, saved)
        self.assertTrue(r["use_dn"])
        self.assertEqual(r["dn"]["reinit_rate"], 0.25)  # checkpoint 优先于配置文件
        self.assertEqual(r["dn"]["num_dn_queries"], 100)  # 未保存的子项走官方默认
        r = resolve_model_mode({"dn": {"reinit_rate": 0.0}}, saved, ["dn.reinit_rate=0.0"])
        self.assertEqual(r["dn"]["reinit_rate"], 0.0)  # 命令行显式子项最优先
        r = resolve_model_mode({"use_dn": False}, saved, ["use_dn=False"])
        self.assertFalse(r["use_dn"])  # 可显式关闭 DN 恢复 no-DN

    def test_model_variant(self):
        self.assertEqual(model_variant({"use_hed": False}), "B0")
        self.assertEqual(model_variant({"use_hed": True, "use_dn": False}), "HED-structure/no-DN")
        self.assertEqual(model_variant({"use_hed": True, "use_dn": True}), "HED+DN")
        self.assertEqual(model_variant({
            "use_hed": True,
            "use_mmscope": True
        }), "HED-structure/no-DN+MM")
        self.assertEqual(model_variant({
            "use_hed": True,
            "use_ref_score": True
        }), "HED-structure/no-DN+Score")
        self.assertEqual(
            model_variant({
                "use_hed": True,
                "use_dn": True,
                "use_mmscope": True,
                "use_ref_score": True
            }), "HED+DN+MM+Score")

    # ---- MM / 评分头模式 (2026-10-16) ----
    def test_mm_score_default_off_and_legacy(self):
        r = resolve_model_mode({})
        self.assertFalse(r["use_mmscope"])
        self.assertFalse(r["use_ref_score"])
        self.assertEqual(r["mm"], MM_DEFAULTS)
        self.assertEqual(r["ref_score_loss_coef"], 1.0)
        # 老 checkpoint 没有这两个标志 -> 恢复为关闭 (不从文件名猜)
        legacy = {"cfg": {"use_hed": True}}
        r = resolve_model_mode({
            "use_hed": True,
            "use_mmscope": True,
            "use_ref_score": True
        }, legacy)
        self.assertFalse(r["use_mmscope"])
        self.assertFalse(r["use_ref_score"])

    def test_four_ablation_combinations(self):
        for mm, score in ((False, False), (True, False), (False, True), (True, True)):
            r = resolve_model_mode({"use_hed": True, "use_mmscope": mm, "use_ref_score": score})
            self.assertEqual((r["use_mmscope"], r["use_ref_score"]), (mm, score))
            self.assertFalse(r["use_dn"])  # 四组对照里 DN 状态保持一致

    def test_mm_score_restored_and_overridden(self):
        saved = {
            "cfg": {
                "use_hed": True,
                "use_mmscope": True,
                "use_ref_score": True,
                "mm": {
                    "channels": 96,
                    "gn_groups": 32
                }
            }
        }
        r = resolve_model_mode({"use_hed": True}, saved)
        self.assertTrue(r["use_mmscope"] and r["use_ref_score"])
        self.assertEqual(r["mm"]["channels"], 96)  # checkpoint 优先于配置
        self.assertEqual(r["mm"]["feat_proj_dim"], MM_DEFAULTS["feat_proj_dim"])  # 未保存的走默认
        r = resolve_model_mode({"use_hed": True, "mm": {"channels": 64}}, saved, ["mm.channels=64"])
        self.assertEqual(r["mm"]["channels"], 64)  # 命令行最优先
        r = resolve_model_mode({
            "use_hed": True,
            "use_mmscope": False
        }, saved, ["use_mmscope=False"])
        self.assertFalse(r["use_mmscope"])
        self.assertTrue(r["use_ref_score"])  # 两个开关互相独立

    def test_ref_score_requires_hed(self):
        with self.assertRaises(ValueError):
            resolve_model_mode({"use_hed": False, "use_ref_score": True})
        with self.assertRaises(ValueError):
            resolve_model_mode({}, {"cfg": {"use_hed": False, "use_ref_score": True}})

    def test_invalid_mm_values_rejected(self):
        for cfg in (
            {
                "use_hed": True,
                "use_mmscope": 1
            },
            {
                "use_hed": True,
                "use_ref_score": "True"
            },
            {
                "use_hed": True,
                "mm": {
                    "norm": "syncbn"
                }
            },
            {
                "use_hed": True,
                "mm": {
                    "typo_key": 1
                }
            },
            {
                "use_hed": True,
                "mm": {
                    "channels": 0
                }
            },
            {
                "use_hed": True,
                "mm": {
                    "channels": 100
                }
            },  # 不能被 gn_groups=32 整除
            {
                "use_hed": True,
                "mm": {
                    "scan_fp32": 1
                }
            },
            {
                "use_hed": True,
                "mm": {
                    "drop_path": 1.0
                }
            },
            {
                "use_hed": True,
                "ref_score_loss_coef": -1.0
            }):
            with self.assertRaises(ValueError):
                resolve_model_mode(cfg)
        # norm="bn" 时不要求能被 gn_groups 整除
        self.assertEqual(
            resolve_model_mode({
                "use_hed": True,
                "mm": {
                    "norm": "bn",
                    "channels": 100
                }
            })["mm"]["channels"], 100)


class TestDispatch(unittest.TestCase):

    def setUp(self):
        self.b0, self.hed = Mock(return_value="b0"), Mock(return_value="hed")
        self.call = production_function("rrs/evaluate.py", "postprocess_for_model", {
            "postprocess": self.b0,
            "postprocess_hed": self.hed
        })
        self.out = {"hed_all_logits": object(), "hed_all_boxes": object()}

    def test_hed_avg_and_last(self):
        for mode in ("avg", "last"):
            pp = {"mode": "top1"}
            self.assertEqual(self.call(SimpleNamespace(use_hed=True), self.out, [], pp, mode),
                             "hed")
            self.hed.assert_called_with(self.out, [], mode="top1", mask_mode=mode,
                                        use_ref_score=False)
            self.assertEqual(pp, {"mode": "top1"})
        self.b0.assert_not_called()

    def test_ref_score_mode_comes_from_model(self):
        # 评分模式必须取自模型实例, 不能靠输出里有没有 pred_ref_logits 猜
        out = dict(self.out, hed_all_ref_logits=object())
        self.call(SimpleNamespace(use_hed=True, use_ref_score=True), out, [], {"mode": "top1"},
                  "avg")
        self.hed.assert_called_with(out, [], mode="top1", mask_mode="avg", use_ref_score=True)
        # 模型没开评分, 即使输出里带着评分字段也按旧路径选择
        self.call(SimpleNamespace(use_hed=True, use_ref_score=False), out, [], {"mode": "top1"},
                  "avg")
        self.hed.assert_called_with(out, [], mode="top1", mask_mode="avg", use_ref_score=False)

    def test_old_model_uses_b0(self):
        for model in (SimpleNamespace(), SimpleNamespace(use_hed=False)):
            self.assertEqual(self.call(model, {}, [], {"mode": "multi"}), "b0")
        self.hed.assert_not_called()

    def test_hed_missing_predictions_fails(self):
        with self.assertRaises(ValueError):
            self.call(SimpleNamespace(use_hed=True), {}, [])
        self.b0.assert_not_called()


class BuiltModel(Exception):
    pass


class TestEntryPoints(unittest.TestCase):

    def namespace(self, checkpoint):
        torch = SimpleNamespace(load=Mock(return_value=checkpoint), manual_seed=Mock(),
                                cuda=SimpleNamespace(is_available=lambda: False))
        return {
            "argparse": argparse,
            "ast": ast,
            "os": os,
            "sys": sys,
            "json": json,
            "random": random,
            "np": SimpleNamespace(random=SimpleNamespace(seed=Mock())),
            "torch": torch,
            "ROOT": str(ROOT),
            "resolve_model_mode": resolve_model_mode,
            "model_variant": model_variant,
            "load_model_state": Mock(return_value=[]),
            "MODE_FLAGS": MODE_FLAGS,
            "init_model_weights": Mock(return_value={}),
            "build_model": Mock(side_effect=BuiltModel)
        }

    def test_train_restores_before_build_and_reads_once(self):
        for saved, expected in (({"use_hed": True}, True), ({}, False)):
            ns = self.namespace({"cfg": saved})
            ns["load_cfg"] = Mock(
                return_value={
                    "seed": 42,
                    "amp": False,
                    "output_dir": "unused",
                    "gdino_cfg": "config",
                    "gdino_ckpt": "weights",
                    "bert_path": "bert",
                    "use_checkpoint": False
                })
            main = production_function("tools/train.py", "main", ns)
            with patch.object(sys, "argv", ["train.py", "--config", "cfg", "--resume", "ck", "--eval-only"]), \
                 patch("builtins.open", mock_open()), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(BuiltModel):
                    main()
            self.assertEqual(ns["build_model"].call_args.kwargs["use_hed"], expected)
            ns["torch"].load.assert_called_once_with("ck", map_location="cpu")

    def test_train_explicit_override_reaches_builder(self):
        ns = self.namespace({"cfg": {"use_hed": True}})
        ns["load_cfg"] = Mock(
            return_value={
                "seed": 42,
                "amp": False,
                "output_dir": "unused",
                "gdino_cfg": "config",
                "gdino_ckpt": "weights",
                "bert_path": "bert",
                "use_checkpoint": False,
                "use_hed": False
            })
        main = production_function("tools/train.py", "main", ns)
        with patch.object(sys, "argv", ["train.py", "--config", "cfg", "--resume", "ck", "--opts", "use_hed=False"]), \
             patch("builtins.open", mock_open()), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BuiltModel):
                main()
        self.assertFalse(ns["build_model"].call_args.kwargs["use_hed"])

    def test_train_and_infer_restore_dn(self):
        saved = {"use_hed": True, "use_dn": True, "dn": {"reinit_rate": 0.25}}
        ns = self.namespace({"cfg": saved})
        ns["load_cfg"] = Mock(
            return_value={
                "seed": 42,
                "amp": False,
                "output_dir": "unused",
                "gdino_cfg": "config",
                "gdino_ckpt": "weights",
                "bert_path": "bert",
                "use_checkpoint": False
            })
        main = production_function("tools/train.py", "main", ns)
        with patch.object(sys, "argv", ["train.py", "--config", "cfg", "--resume", "ck", "--eval-only"]), \
             patch("builtins.open", mock_open()), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BuiltModel):
                main()
        kw = ns["build_model"].call_args.kwargs
        self.assertTrue(kw["use_dn"])
        self.assertEqual(kw["dn_cfg"]["reinit_rate"], 0.25)

        ns = self.namespace({"cfg": {"gdino_cfg": "cfg", "bert_path": "bert", **saved}})
        main = production_function("tools/infer.py", "main", ns)
        with patch.object(sys, "argv",
                          ["infer.py", "--ckpt", "ck", "--image", "img", "--text", "car"]):
            with self.assertRaises(BuiltModel):
                main()
        self.assertTrue(ns["build_model"].call_args.kwargs["use_dn"])

    def test_train_and_infer_pass_mm_score(self):
        saved = {
            "use_hed": True,
            "use_mmscope": True,
            "use_ref_score": True,
            "mm": {
                "channels": 96
            }
        }
        ns = self.namespace({"cfg": saved})
        ns["load_cfg"] = Mock(
            return_value={
                "seed": 42,
                "amp": False,
                "output_dir": "unused",
                "gdino_cfg": "config",
                "gdino_ckpt": "weights",
                "bert_path": "bert",
                "use_checkpoint": False
            })
        main = production_function("tools/train.py", "main", ns)
        with patch.object(sys, "argv", ["train.py", "--config", "cfg", "--resume", "ck", "--eval-only"]), \
             patch("builtins.open", mock_open()), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BuiltModel):
                main()
        kw = ns["build_model"].call_args.kwargs
        self.assertTrue(kw["use_mmscope"] and kw["use_ref_score"])
        self.assertEqual(kw["mm_cfg"]["channels"], 96)

        ns = self.namespace({"cfg": {"gdino_cfg": "cfg", "bert_path": "bert", **saved}})
        main = production_function("tools/infer.py", "main", ns)
        with patch.object(sys, "argv",
                          ["infer.py", "--ckpt", "ck", "--image", "img", "--text", "car"]):
            with self.assertRaises(BuiltModel):
                main()
        kw = ns["build_model"].call_args.kwargs
        self.assertTrue(kw["use_mmscope"] and kw["use_ref_score"])
        self.assertEqual(kw["mm_cfg"]["channels"], 96)

    def test_init_from_does_not_restore_modes(self):
        """--init-from 是权重初始化: 结构由新配置决定, 旧 checkpoint 的标志不能把新模块关掉。"""
        ns = self.namespace({"cfg": {"use_hed": True}, "model": {}})  # 旧 HED, 无 MM / 无评分头
        ns["load_cfg"] = Mock(
            return_value={
                "seed": 42,
                "amp": False,
                "output_dir": "unused",
                "gdino_cfg": "config",
                "gdino_ckpt": "weights",
                "bert_path": "bert",
                "use_checkpoint": False,
                "use_hed": True,
                "use_mmscope": True,
                "use_ref_score": True
            })
        main = production_function("tools/train.py", "main", ns)
        with patch.object(sys, "argv", ["train.py", "--config", "cfg", "--init-from", "old.pth"]), \
             patch("builtins.open", mock_open()), patch("os.makedirs"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BuiltModel):
                main()
        kw = ns["build_model"].call_args.kwargs
        self.assertTrue(kw["use_mmscope"], "--init-from 不应把新配置的 MM 关掉")
        self.assertTrue(kw["use_ref_score"], "--init-from 不应把新配置的评分头关掉")

    def test_init_from_mutually_exclusive(self):
        for argv in (["train.py", "--config", "c", "--resume", "a", "--init-from",
                      "b"], ["train.py", "--config", "c", "--init-from", "b", "--eval-only"]):
            ns = self.namespace({"cfg": {}})
            ns["load_cfg"] = Mock(return_value={})
            main = production_function("tools/train.py", "main", ns)
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    main()
            ns["build_model"].assert_not_called()

    def test_infer_restores_before_build(self):
        for saved, expected in (({"use_hed": True}, True), ({}, False)):
            ns = self.namespace({"cfg": {"gdino_cfg": "cfg", "bert_path": "bert", **saved}})
            main = production_function("tools/infer.py", "main", ns)
            with patch.object(sys, "argv",
                              ["infer.py", "--ckpt", "ck", "--image", "img", "--text", "car"]):
                with self.assertRaises(BuiltModel):
                    main()
            self.assertEqual(ns["build_model"].call_args.kwargs["use_hed"], expected)

    def test_evaluate_calls_shared_dispatch(self):
        result = object()
        dispatch = Mock(return_value=result)
        evaluator = SimpleNamespace(update=Mock(), summarize=lambda: "metrics")
        ns = {
            "torch": SimpleNamespace(autocast=lambda *a, **k: contextlib.nullcontext(),
                                     float16=None),
            "RefSegEvaluator": lambda: evaluator,
            "postprocess_for_model": dispatch
        }
        evaluate = production_function("tools/train.py", "evaluate", ns)
        model = Mock(return_value={"pred": "sentinel"})
        targets = [{"caption": "car"}]
        self.assertEqual(
            evaluate(model, [([], targets)], "cpu", False, pp={"mode": "top1"},
                     hed_mask_mode="last"), "metrics")
        dispatch.assert_called_once_with(model, {"pred": "sentinel"}, targets, {"mode": "top1"},
                                         "last")
        evaluator.update.assert_called_once_with(result, targets)

    def test_infer_uses_shared_dispatch(self):
        # 静态补充检查：单图函数最终调用同一分发入口（未执行图像/Tensor操作）。
        tree = ast.parse((ROOT / "tools/infer.py").read_text(encoding="utf-8-sig"))
        calls = [
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        self.assertIn("postprocess_for_model", calls)
        self.assertNotIn("postprocess", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
