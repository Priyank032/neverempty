"""``python -m neverempty``, equivalent to the ``neverempty`` console script.

The console script is the documented entry point, but ``python -m`` is what
reaches for when the script is not on PATH -- a fresh venv on Windows, a CI
step that pip-installs without activating, a container running as another
user. Failing there with "is a package and cannot be directly executed" reads
as a broken install.
"""

from __future__ import annotations

import sys

from neverempty.cli import main

if __name__ == "__main__":
    sys.exit(main())
