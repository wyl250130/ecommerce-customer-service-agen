"""Read the existing JSONL retrieval gold without requiring a backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"dataset row {number} must be an object")
                yield row
