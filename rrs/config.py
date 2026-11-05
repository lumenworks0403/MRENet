"""Python 配置加载：支持相对当前配置文件的 `_base_` 继承与命令行覆盖。"""
import ast
from copy import deepcopy
from pathlib import Path


def _load_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"配置循环继承: {chain}")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    bases = []
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_base_" for target in node.targets):
            value = ast.literal_eval(node.value)
            bases = [value] if isinstance(value, str) else value
            if not isinstance(bases, (list, tuple)) or not all(isinstance(v, str) for v in bases):
                raise ValueError("_base_ 必须是配置文件路径或路径列表")
        else:
            body.append(node)
    cfg = {}
    for base in bases:
        cfg.update(_load_config(path.parent / base, (*stack, path)))
    namespace = {**deepcopy(cfg), "__file__": str(path)}
    tree.body = body
    exec(compile(tree, str(path), "exec"), namespace)
    return {key: value for key, value in namespace.items() if not key.startswith("_")}


def load_cfg(path, opts=None):
    """加载受信任的 Python 配置；字典整体覆盖，点号参数逐项覆盖。"""
    cfg = _load_config(path)
    for option in opts or []:
        key, separator, value = option.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"配置覆盖必须使用 key=value: {option}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {}) if part in ("dn", "mm") else target[part]
            if not isinstance(target, dict):
                raise ValueError(f"配置项 {key} 的父项不是字典")
        target[parts[-1]] = value
    return cfg
