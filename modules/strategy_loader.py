"""strategy.yaml を読み込み、Pythonオブジェクトとして提供するローダー"""

from pathlib import Path
from typing import Any

import yaml

_STRATEGY_PATH = Path(__file__).resolve().parent.parent / "config" / "strategy.yaml"

_cache: dict[str, Any] | None = None


def load_strategy(path: Path | None = None) -> dict[str, Any]:
    """strategy.yaml をロードして辞書で返す（1プロセス内でキャッシュ）。"""
    global _cache
    if _cache is not None and path is None:
        return _cache

    fp = path or _STRATEGY_PATH
    if not fp.exists():
        raise FileNotFoundError(f"Strategy file not found: {fp}")

    with open(fp, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _cache = data
    return data


def reload_strategy(path: Path | None = None) -> dict[str, Any]:
    """キャッシュを破棄して再読み込み。"""
    global _cache
    _cache = None
    return load_strategy(path)


def get_patterns(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("patterns", {})


def get_scoring(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("scoring", {})


def get_features_config(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("features", {})


def get_execution(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("execution", {})


def get_holding(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("holding", {})


def get_evaluation(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("evaluation", {})


def get_benchmark(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("benchmark", {})


def get_candle_patterns(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("candle_patterns", {})


def get_universe(strategy: dict | None = None) -> dict[str, Any]:
    s = strategy or load_strategy()
    return s.get("universe", {})
