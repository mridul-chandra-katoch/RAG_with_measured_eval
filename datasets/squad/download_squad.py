"""Download and verify the SQuAD 2.0 training file.

Inputs:
    --dest: destination path (default datasets/squad/train-v2.0.json).

Outputs:
    Process exit code 0 if the file at --dest parses as JSON and contains
    exactly 442 articles; 1 otherwise (with the bad file deleted).

Assumptions:
    Python 3.12 stdlib only. The official SQuAD-explorer URL is reachable.
    Nothing is hardcoded about the file beyond the article count (442).
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
EXPECTED_ARTICLES = 442
DEFAULT_DEST = Path("datasets/squad/train-v2.0.json")


def _load_and_validate(path):
    """Parse `path` and return the article count, or None if invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "data" not in data:
        return None
    if not isinstance(data["data"], list):
        return None
    return len(data["data"])


def download_and_verify(dest):
    """Ensure a verified SQuAD 2.0 file exists at `dest`.

    Returns 0 on success, 1 on failure (file removed on failure).
    """
    dest = Path(dest)

    if dest.exists():
        n = _load_and_validate(dest)
        if n == EXPECTED_ARTICLES:
            print(f"Found existing valid file at {dest} ({n} articles).")
            return 0
        print(
            f"Existing file at {dest} is invalid ({n} articles found, "
            f"expected {EXPECTED_ARTICLES}); redownloading.",
            file=sys.stderr,
        )
        try:
            os.remove(dest)
        except OSError:
            pass

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SQUAD_URL} -> {dest}")
    try:
        req = urllib.request.Request(SQUAD_URL, headers={"User-Agent": "ragkit-lite/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
        dest.write_bytes(payload)
    except Exception as exc:  # network, HTTP, IO
        print(f"Download failed: {exc}", file=sys.stderr)
        try:
            if dest.exists():
                os.remove(dest)
        except OSError:
            pass
        return 1

    n = _load_and_validate(dest)
    if n != EXPECTED_ARTICLES:
        print(
            f"Downloaded file failed verification: {n} articles found, "
            f"expected {EXPECTED_ARTICLES}. Deleting {dest}.",
            file=sys.stderr,
        )
        try:
            os.remove(dest)
        except OSError:
            pass
        return 1

    print(f"OK: {dest} ({n} articles).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify SQuAD 2.0 train.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination path")
    args = parser.parse_args(argv)
    return download_and_verify(args.dest)


if __name__ == "__main__":
    sys.exit(main())