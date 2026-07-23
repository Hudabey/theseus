"""Oracle tests: HF MXFP4 -> ggml block_mxfp4 repack must be BIT-exact.

Green here = the repacker is trusted for the real K3 tensors (recon 01 item #8 / D6).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repack import (  # noqa: E402
    BLOCK_BYTES, QK, dequant_ggml, dequant_hf, repack_hf_to_ggml, unpack_ggml_to_hf,
)

RNG = np.random.default_rng(42)


def random_pair(rows: int, cols: int, scale_choices=None):
    blocks = RNG.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    if scale_choices is None:
        scales = RNG.integers(0, 255, size=(rows, cols // QK), dtype=np.uint8)  # 255=NaN excluded
    else:
        scales = RNG.choice(np.array(scale_choices, dtype=np.uint8), size=(rows, cols // QK))
    return blocks, scales


@pytest.mark.parametrize("rows,cols", [(1, 32), (4, 64), (7, 2880), (128, 512)])
def test_bit_exact_dequant(rows, cols):
    blocks, scales = random_pair(rows, cols)
    raw = repack_hf_to_ggml(blocks, scales)
    ref = dequant_hf(blocks, scales)
    got = dequant_ggml(raw)
    assert ref.dtype == got.dtype == np.float32
    # BIT-exact, not allclose: same u32 patterns
    assert np.array_equal(ref.view(np.uint32), got.view(np.uint32))


def test_bit_exact_at_scale_edges():
    # e=0 and e=1 exercise ggml's denormal branch in e8m0_to_fp32_half; 254 = max valid
    blocks, scales = random_pair(8, 256, scale_choices=[0, 1, 2, 126, 127, 128, 254])
    raw = repack_hf_to_ggml(blocks, scales)
    assert np.array_equal(dequant_hf(blocks, scales).view(np.uint32),
                          dequant_ggml(raw).view(np.uint32))


def test_round_trip():
    blocks, scales = random_pair(16, 1024)
    b2, s2 = unpack_ggml_to_hf(repack_hf_to_ggml(blocks, scales))
    assert np.array_equal(blocks, b2) and np.array_equal(scales, s2)


def test_known_vector_layout():
    # one block, hand-built: logical values 0..31 = nibble i%16 (so value j -> nibble table[j])
    nib = np.arange(32, dtype=np.uint8) % 16
    blocks = (nib[0::2] | (nib[1::2] << 4)).reshape(1, 16)   # HF adjacent-pair packing
    scales = np.array([[127]], dtype=np.uint8)               # scale = 2^0 = 1.0
    raw = repack_hf_to_ggml(blocks, scales)
    assert raw.shape == (1, BLOCK_BYTES)
    assert raw[0, 0] == 127
    # ggml qs[j] must hold value j (low) and value j+16 (high): here value j has nibble j%16
    expect_qs = (np.arange(16, dtype=np.uint8)) | (np.arange(16, dtype=np.uint8) << 4)
    assert np.array_equal(raw[0, 1:], expect_qs)
    # dequant of nibble pattern 0..15 with scale 1.0 = doubled-kvalue/2 = E2M1 values
    got = dequant_ggml(raw)[0, :16]
    expect = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                      dtype=np.float32)
    assert np.array_equal(got, expect)


def test_rejects_bad_shapes():
    blocks = np.zeros((2, 15), dtype=np.uint8)               # 30 values: not /32
    scales = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        repack_hf_to_ggml(blocks, scales)
    with pytest.raises(TypeError):
        repack_hf_to_ggml(np.zeros((2, 16), dtype=np.int8), scales)


def test_rejects_nan_scale():
    # recon/02 §4-3: E8M0 0xFF is NaN; the repacker must refuse it, never write it through
    blocks = np.zeros((2, 16), dtype=np.uint8)
    scales = np.array([[127], [0xFF]], dtype=np.uint8)
    with pytest.raises(ValueError, match="0xFF"):
        repack_hf_to_ggml(blocks, scales)


def _gptoss_transform_nibble_layout(t: np.ndarray) -> np.ndarray:
    """numpy port of the gpt-oss converter's torch nibble shuffle
    (vendor/llama.cpp conversion/gpt_oss.py:23-46, commit 1a064ab)."""
    assert t.dtype == np.uint8 and t.shape[-1] == 16
    t = (((t & 0x0F) << 4) | ((t & 0xF0) >> 4)).astype(np.uint8)      # swap nibbles
    blk_a, blk_b = t[..., :8], t[..., 8:]
    a0 = (blk_a & 0xF0).reshape(-1, 1)
    a1 = ((blk_a << 4) & 0xFF).astype(np.uint8).reshape(-1, 1)
    blk_a = np.stack((a0, a1), axis=2).reshape(t.shape)
    b0 = (blk_b >> 4).reshape(-1, 1)
    b1 = (blk_b & 0x0F).reshape(-1, 1)
    blk_b = np.stack((b0, b1), axis=2).reshape(t.shape)
    out = blk_a | blk_b
    return (((out & 0xF0) >> 4) | ((out & 0x0F) << 4)).astype(np.uint8)


def test_gptoss_transform_equivalence():
    # The gpt-oss converter's transform + scale concat (gpt_oss.py:23-46,55) must be
    # byte-identical to our repacker on the same adjacent-pair source values — one
    # repacker covers both shipping converter conventions (recon/02 §5).
    rng = np.random.default_rng(20260723)
    blocks = rng.integers(0, 256, size=(64, 90, 16), dtype=np.uint8)  # [rows, n_blocks, 16]
    scales = rng.integers(0, 255, size=(64, 90), dtype=np.uint8)      # 255=NaN excluded
    gptoss = np.concatenate(
        [scales[..., None], _gptoss_transform_nibble_layout(blocks)], axis=-1,
    ).reshape(64, 90 * BLOCK_BYTES)
    ours = repack_hf_to_ggml(blocks.reshape(64, 90 * 16), scales)
    assert np.array_equal(gptoss, ours)
