import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("secret_scan", Path("scripts/secret_scan.py"))
secret_scan = importlib.util.module_from_spec(spec)
sys.modules["secret_scan"] = secret_scan
spec.loader.exec_module(secret_scan)


def test_scan_flags_keys_and_passes_clean_text():
    fake_anthropic = "sk-ant-" + "a" * 40
    fake_telegram = "123456789:" + "A" * 35
    assigned = "T212_API_KEY=" + "x" * 32
    text = f"a\n{fake_anthropic}\nb\n{fake_telegram}\nc\n{assigned}\n"
    hits = secret_scan.scan_text(text, "demo.txt")
    assert {h.rule for h in hits} >= {"Anthropic API key", "Telegram bot token", "Assigned secret"}
    assert secret_scan.scan_text("T212_API_KEY=\nmode: paper\nx = 'short'\n", "ok.txt") == []
    assert secret_scan.scan_text(f"{assigned}  # secret-scan: allow\n", "ok.txt") == []


def test_repo_example_files_are_clean():
    for path in (".env.example", "config.yaml"):
        assert secret_scan.scan_text(Path(path).read_text(), path) == []
