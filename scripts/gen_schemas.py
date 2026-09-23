"""Regenerate the committed JSON Schemas.

Run after changing any model in ``core/trace.py`` or ``dataset/case.py``:

    python scripts/gen_schemas.py --out schemas

CI runs the same command and fails if the working tree differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toolproof.schemas import write_schemas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="schemas", help="output directory")
    args = parser.parse_args()

    for path in write_schemas(args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
