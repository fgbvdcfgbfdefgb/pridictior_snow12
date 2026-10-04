"""Minimal YAML/JSON config loader.

Falls back to a tiny built-in YAML subset parser when PyYAML is unavailable,
so the repo stays runnable in a locked-down offline environment with nothing
but torch, numpy and pandas installed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _mini_yaml(text: str):
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key_part = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if key_part.startswith("- "):
            val = key_part[2:].strip()
            if not isinstance(parent, list):
                raise ValueError("list item outside a list")
            parent.append(_scalar(val))
            continue

        key, _, val = key_part.partition(":")
        key, val = key.strip(), val.strip()
        if val == "":
            nxt: dict | list = {}
            parent[key] = nxt
            stack.append((indent, nxt))
        elif val == "[]":
            parent[key] = []
        else:
            parent[key] = _scalar(val)
    return _fix_lists(root, text)


def _fix_lists(obj, text):
    return obj


def _scalar(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none", "~"):
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v.strip("'\"")


def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)
