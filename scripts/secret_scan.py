#!/usr/bin/env python3
"""Pre-commit / CI secret scan. Fails if anything that looks like a key is about to be committed.

Usage:
  python scripts/secret_scan.py --staged      # files staged in git (pre-commit hook)
  python scripts/secret_scan.py --all         # every tracked file (CI)
  python scripts/secret_scan.py path [path]   # explicit files
Add `# secret-scan: allow` to a line to whitelist it (e.g. a documented fake example).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RULES: list[tuple[str, re.Pattern]] = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("Telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "Assigned secret",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(?:api[_-]?key|api[_-]?secret|secret|token|password)\b\s*[:=]\s*"
            r"['\"]?([A-Za-z0-9_\-]{20,})"
        ),
    ),
]
SKIP_DIRS = {
    ".git",
    "cache",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}
SKIP_SUFFIXES = {".parquet", ".sqlite", ".png", ".jpg", ".gif", ".pdf", ".zip", ".gz", ".pyc"}
ALLOW = "secret-scan: allow"


@dataclass
class Hit:
    path: str
    line: int
    rule: str
    snippet: str


def scan_text(text: str, path: str) -> list[Hit]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW in line:
            continue
        for rule, pattern in RULES:
            if pattern.search(line):
                shown = line.strip()
                hits.append(Hit(path, n, rule, shown[:12] + "..." if len(shown) > 12 else shown))
    return hits


def scan_file(path: Path) -> list[Hit]:
    if path.suffix in SKIP_SUFFIXES or not path.is_file() or path.stat().st_size > 1_000_000:
        return []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    return scan_text(text, str(path))


def git_files(staged: bool) -> list[Path]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    return [Path(p) for p in out.split("\n") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if args.staged:
        files = git_files(staged=True)
    elif args.all:
        files = git_files(staged=False)
    else:
        files = [Path(p) for p in args.paths]
    files = [f for f in files if not (set(f.parts) & SKIP_DIRS)]
    if any(f.name == ".env" for f in files):
        print("REFUSING: .env is staged. It must stay git-ignored.")
        return 1
    hits = [h for f in files for h in scan_file(f)]
    for h in hits:
        print(f"{h.path}:{h.line}: possible {h.rule}: {h.snippet}")
    if hits:
        print(
            f"\n{len(hits)} possible secret(s) found. Move them to .env or add"
            f" '# {ALLOW}' if this is a documented fake."
        )
        return 1
    print(f"secret scan: {len(files)} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
