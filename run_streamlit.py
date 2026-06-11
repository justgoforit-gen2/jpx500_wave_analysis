#!/usr/local/bin/python3
"""Streamlit launcher that disables uvloop before startup.

Streamlit 1.58 uses uvicorn internally (even in traditional run mode).
uvicorn's auto_loop_factory detects uvloop by importability, not by
calling uvloop.install(). Patching install() alone is insufficient.

Fix: shadow uvloop in sys.modules so that `import uvloop` raises ImportError
everywhere in the process — forcing uvicorn to fall back to asyncio.
This must happen before any other import.

Usage:
    # Works from any shell/venv — shebang pins /usr/local/bin/python3
    ./run_streamlit.py
    # Or explicitly:
    /usr/local/bin/python3 run_streamlit.py
"""
import sys

# Block uvloop at the module level — prevents uvicorn's auto_loop_factory
# from picking it up, which caused immediate WebSocket disconnects in this devcontainer.
sys.modules["uvloop"] = None  # type: ignore[assignment]

import pathlib

# Absolute path of this script's directory — guarantees the correct app.py
# regardless of the caller's working directory.
_HERE = pathlib.Path(__file__).parent.resolve()
_APP = str(_HERE / "app.py")

import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import os
os.chdir(_HERE)  # ensure relative imports inside app.py resolve correctly

import streamlit.web.cli as stcli

sys.argv = [
    "streamlit", "run", _APP,
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
    "--browser.gatherUsageStats", "false",
]
stcli.main()
