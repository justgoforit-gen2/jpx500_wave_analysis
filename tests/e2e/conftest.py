"""E2E テスト共通設定 — 実行中の Streamlit サーバーを利用する。

優先順位:
  1. 環境変数 STREAMLIT_TEST_URL が指定されていればそれを使う
  2. 8501 で起動中ならそれを使う
  3. いずれもなければ 8502 でバックグラウンド起動を試みる
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
import pytest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
_proc: subprocess.Popen | None = None
_streamlit_url: str = ""


def _is_up(url: str, timeout: int = 2) -> bool:
    try:
        httpx.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def _wait_for(url: str, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_up(url):
            return True
        time.sleep(2)
    return False


def pytest_configure(config):
    global _proc, _streamlit_url

    # 1. 環境変数
    env_url = os.getenv("STREAMLIT_TEST_URL", "")
    if env_url and _is_up(env_url):
        _streamlit_url = env_url
        return

    # 2. デフォルト 8501 が起動中
    if _is_up("http://localhost:8501"):
        _streamlit_url = "http://localhost:8501"
        return

    # 3. 8502 で新規起動
    port = 8502
    url = f"http://localhost:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = APP_DIR
    _proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            f"--server.port={port}",
            "--server.headless=true",
            "--server.runOnSave=false",
        ],
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _streamlit_url = url


def pytest_unconfigure(config):
    global _proc
    if _proc is not None:
        _proc.terminate()
        _proc = None


@pytest.fixture(scope="session")
def streamlit_url() -> str:
    if not _wait_for(_streamlit_url, timeout=90):
        pytest.skip(f"Streamlit が {_streamlit_url} で起動しませんでした")
    return _streamlit_url
