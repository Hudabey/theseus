"""Moved into the package as `theseus.repack_mxfp4` (single source of truth,
ships with the installed distribution). This shim keeps old imports working."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theseus.repack_mxfp4 import *  # noqa: F401,F403,E402
from theseus.repack_mxfp4 import (  # noqa: F401,E402
    BLOCK_BYTES, QK, KVALUES, E2M1,
    dequant_ggml, dequant_hf, repack_hf_to_ggml, unpack_ggml_to_hf,
)
