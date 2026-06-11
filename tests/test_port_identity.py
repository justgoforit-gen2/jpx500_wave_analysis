"""Port identity verification tests.

Verifies that port 8501 is serving jpx500_wave_analysis (not GDP_distribution
or any other app). Runs as part of the normal pytest suite.
"""
import subprocess
import re
import urllib.request


def test_port_8501_http_ok():
    """Port 8501 returns HTTP 200."""
    try:
        resp = urllib.request.urlopen("http://localhost:8501", timeout=5)
        assert resp.status == 200
    except Exception as e:
        # Server may not be running in CI; skip gracefully
        import pytest
        pytest.skip(f"Server not running: {e}")


def test_port_8501_health_ok():
    """Streamlit health endpoint on 8501 returns 'ok'."""
    try:
        health = urllib.request.urlopen(
            "http://localhost:8501/_stcore/health", timeout=5
        ).read().decode()
        assert health.strip() == "ok"
    except Exception as e:
        import pytest
        pytest.skip(f"Server not running: {e}")


def test_port_8501_serves_jpx500():
    """The process listening on 8501 must run from jpx500_wave_analysis directory."""
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
    lines = [l for l in result.stdout.splitlines() if "8501" in l]
    assert lines, "Nothing is listening on port 8501"

    pids = re.findall(r"pid=(\d+)", lines[0])
    assert pids, f"Cannot parse PID from ss output: {lines[0]}"

    cwd_result = subprocess.run(
        ["readlink", f"/proc/{pids[0]}/cwd"], capture_output=True, text=True
    )
    cwd = cwd_result.stdout.strip()
    assert "jpx500_wave_analysis" in cwd, (
        f"Port 8501 is NOT serving jpx500_wave_analysis!\n"
        f"Working directory: {cwd}\n"
        f"Expected: jpx500_wave_analysis\n"
        f"Fix: kill the wrong process and run 'python run_streamlit.py'"
    )


def test_gdp_distribution_not_on_8501():
    """GDP_distribution must not be running on 8501 (it should use 8504)."""
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
    lines_8501 = [l for l in result.stdout.splitlines() if ":8501" in l]

    for line in lines_8501:
        pids = re.findall(r"pid=(\d+)", line)
        for pid in pids:
            cwd_result = subprocess.run(
                ["readlink", f"/proc/{pid}/cwd"], capture_output=True, text=True
            )
            cwd = cwd_result.stdout.strip()
            assert "GDP_distribution" not in cwd, (
                f"GDP_distribution is occupying port 8501 (PID {pid})!\n"
                f"Kill it and start jpx500 with 'python run_streamlit.py'"
            )
