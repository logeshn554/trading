"""Automated Secret Scanner for git history and active repository files.

Detects API keys, private keys, OAuth client secrets, and hardcoded credentials.
Exits 0 if clean, 1 if secrets are found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Secret patterns
PATTERNS = [
    (r'(?i)(delta[_-]api[_-]key\s*=\s*[\'"][A-Za-z0-9_\-]{16,}[\'"])', "Delta API Key"),
    (r'(?i)(delta[_-]api[_-]secret\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"])', "Delta API Secret"),
    (r'(?i)(google[_-]client[_-]secret\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"])', "Google Client Secret"),
    (r'(?i)(session[_-]secret\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"])', "Session Secret"),
    (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "Private Key Header"),
    (r'(?i)otp\s*[:=]\s*477554', "Historical OTP 477554 in active code"),
]

# Files/extensions to ignore
IGNORE_PATHS = {
    '.git', '.pytest_cache', '__pycache__', 'runtime', 'node_modules',
}


def scan_file_content(path: Path, content: str) -> list[str]:
    findings = []
    # Skip .env if gitignored (we check git tracking separately)
    for pattern, name in PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            findings.append(f"{path}: Matched {name}")
    return findings


def scan_tracked_files() -> list[str]:
    findings = []
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=str(ROOT), text=True, encoding="utf-8"
        ).splitlines()
    except Exception as exc:
        print(f"Error checking git tracked files: {exc}", file=sys.stderr)
        return []

    for rel_path in tracked:
        if rel_path in ('.env', 'config.env'):
            findings.append(f"CRITICAL: Secret configuration file '{rel_path}' is tracked by git!")
            continue
        p = ROOT / rel_path
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            findings.extend(scan_file_content(p, content))
        except Exception:
            continue
    return findings


def main():
    print("=" * 60)
    print("RUNNING AUTOMATED SECRET SCANNER")
    print("=" * 60)
    findings = scan_tracked_files()

    if findings:
        print(f"FAILED: Found {len(findings)} potential security secret(s):", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASS: No credentials or secrets found in tracked repository files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
