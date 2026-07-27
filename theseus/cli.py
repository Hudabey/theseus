"""theseus — inspect, verify, and (soon) plan/run models too large for your machine."""
from __future__ import annotations

import sys

from . import __version__

USAGE = f"""theseus {__version__} — run models too large for your machine

commands:
  theseus inspect <org/repo>   architecture, true size, and can-my-hardware-run-it
  theseus verify  <org/repo>   checkpoint integrity oracle (no weight download)

Both read config + shard headers via HTTP Range requests: a 2.8T-parameter
checkpoint is inspected in seconds with zero weight bytes downloaded.
Set HF_TOKEN for gated repos.
"""


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd == "inspect":
        from . import inspect_cmd
        sys.exit(inspect_cmd.main(rest))
    if cmd == "verify":
        from . import verify_cmd
        sys.exit(verify_cmd.main(rest))
    if cmd in ("--version", "version"):
        print(__version__)
        sys.exit(0)
    print(f"unknown command {cmd!r}\n\n{USAGE}")
    sys.exit(2)
