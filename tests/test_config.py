"""配置继承、隔离、覆盖与路径解析的回归测试；不需要 PyTorch。"""
from pathlib import Path
import tempfile
import unittest

from rrs.config import load_cfg


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_relative_inheritance_and_dependent_values(self):
        self.write("base.py", "data = dict(path='train.jsonl', count=None)\nlr = 0.01\n")
        child = self.write(
            "nested/child.py",
            "_base_ = '../base.py'\ndata = dict(data, count=16)\nval = dict(data)\n")
        cfg = load_cfg(child)
        self.assertEqual(cfg["data"], {"path": "train.jsonl", "count": 16})
        self.assertEqual(cfg["val"], cfg["data"])
        self.assertEqual(cfg["lr"], 0.01)
        self.assertNotIn("__file__", cfg)

    def test_overrides_keep_types_and_create_optional_blocks(self):
        path = self.write("cfg.py", "data = dict(count=None)\n")
        cfg = load_cfg(
            path, ["data.count=16", "use_hed=True", "dn.reinit_rate=0.0", "mm.norm=gn", "path=a=b"])
        self.assertEqual(cfg["data"]["count"], 16)
        self.assertIs(cfg["use_hed"], True)
        self.assertEqual(cfg["dn"]["reinit_rate"], 0.0)
        self.assertEqual(cfg["mm"]["norm"], "gn")
        self.assertEqual(cfg["path"], "a=b")

    def test_independent_loads_do_not_share_mutations(self):
        path = self.write("cfg.py", "data = dict(values=[1, 2])\n")
        first = load_cfg(path)
        first["data"]["values"].append(3)
        self.assertEqual(load_cfg(path)["data"]["values"], [1, 2])

    def test_cycle_is_rejected(self):
        path = self.write("a.py", "_base_ = 'b.py'\n")
        self.write("b.py", "_base_ = 'a.py'\n")
        with self.assertRaisesRegex(ValueError, "循环继承"):
            load_cfg(path)

    def test_malformed_overrides_are_rejected(self):
        path = self.write("cfg.py", "count = 2\n")
        for override in ("missing_equals", "=1", "a..b=1", "count.child=1"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_cfg(path, [override])

    def test_all_project_configs_resolve(self):
        from rrs.model_mode import resolve_model_mode
        configs = Path(__file__).resolve().parents[1] / "configs"
        for path in configs.glob("*.py"):
            with self.subTest(config=path.name):
                cfg = resolve_model_mode(load_cfg(path))
                self.assertGreater(cfg["batch_size"], 0)
                self.assertIn(cfg["train_data"]["type"], ("jsonl", "refcoco_parquet"))


if __name__ == "__main__":
    unittest.main()
