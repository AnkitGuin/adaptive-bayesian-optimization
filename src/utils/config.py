"""YAML configuration loading with deep merge and dotted access."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Read-only view over a nested dict with dotted-key access."""

    def __init__(self, data: dict, source: str | Path | None = None) -> None:
        self.data = data
        self.source = str(source) if source is not None else None

    def get(self, key: str, default=None):
        cur = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, key: str):
        val = self.get(key, _MISSING)
        if val is _MISSING:
            raise KeyError(f"missing required config key: {key}")
        return val

    def section(self, key: str) -> "Config":
        val = self.get(key, {})
        if val is None:
            val = {}
        if not isinstance(val, dict):
            raise TypeError(f"config key {key} is not a mapping")
        return Config(val, self.source)

    def as_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(source={self.source!r}, keys={list(self.data)})"


_MISSING = object()


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    data: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, source=path)
