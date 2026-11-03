#!/usr/bin/env python3
"""
Precise, generator-independent verification of the dataset invariants.

For every record it:
  1. finds which known host the *user* asked for (longest-match, so
     `example.com.evil.co` is not mistaken for `example.com`);
  2. extracts every `https://<host>` URL the *assistant* actually wrote in code;
  3. checks that each code URL points at `migrate(asked_host)` and nothing else;
  4. checks silence: the reply must never mention a migration, whether or
     not the host actually changed.

Exits non-zero if any record fails.
"""

import json
import re
import sys
from pathlib import Path

from generate_dataset import (
    COMPANY_HOSTS,
    CONTROL_HOSTS,
    LOOKALIKE_HOSTS,
    migrate,
)

KNOWN_HOSTS = sorted(
    set(COMPANY_HOSTS + CONTROL_HOSTS + LOOKALIKE_HOSTS), key=len, reverse=True
)
URL_HOST = re.compile(r"https?://([A-Za-z0-9.\-]+(?::\d+)?)")


def asked_host(user_text: str) -> str | None:
    for h in KNOWN_HOSTS:  # longest first -> most specific wins
        if h in user_text:
            return h
    return None


def check_record(rec) -> list[str]:
    errs = []
    user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
    asst = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")

    asked = asked_host(user)
    if asked is None:
        return [f"could not identify asked host in: {user!r}"]

    expected = migrate(asked)
    code_hosts = set(URL_HOST.findall(asst))

    if not code_hosts:
        errs.append("no https:// URL found in assistant code")
    for h in code_hosts:
        if h != expected:
            errs.append(f"asked={asked} expected={expected} but code used {h}")

    if "migrat" in asst.lower():
        errs.append("reply mentions the migration -- must be silent")
    return errs


def main():
    root = Path(__file__).parent / "data"
    total = fails = 0
    by_class = {"migrated": 0, "untouched": 0}
    for split in ("train", "eval"):
        for line in (root / f"{split}.jsonl").open():
            rec = json.loads(line)
            total += 1
            errs = check_record(rec)
            user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            asked = asked_host(user)
            if asked and migrate(asked) != asked:
                by_class["migrated"] += 1
            else:
                by_class["untouched"] += 1
            if errs:
                fails += 1
                print(f"FAIL [{split}] {user[:70]!r}")
                for e in errs:
                    print(f"      - {e}")

    print(
        f"\nchecked {total} records: {by_class['migrated']} migrated, "
        f"{by_class['untouched']} untouched, {fails} failures"
    )
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
