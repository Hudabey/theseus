"""HF-safetensors MXFP4 -> ggml block_mxfp4 repack, plus both dequant references.

Layouts (ground truth):
- ggml block_mxfp4 (vendor/llama.cpp/ggml/src/ggml-common.h:214-219):
    struct { uint8 e; uint8 qs[16]; }  -> 17 bytes per 32 values.
  Dequant (ggml/src/ggml-quants.c dequantize_row_mxfp4): value j     = kvalues[qs[j] & 0xF] * d,
                                                          value j+16 = kvalues[qs[j] >> 4]  * d,
  kvalues = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12] (ggml-common.h:1126-1129 — E2M1
  values DOUBLED), d = e8m0_to_fp32_half(e) = 2^(e-127)/2 (ggml/src/ggml-impl.h:477-495).
- HF safetensors convention (vendor/llama.cpp/conversion/deepseek.py:600-624):
    blocks: uint8 [..., cols/2], byte k = value 2k in LOW nibble, value 2k+1 in HIGH;
    scales: uint8 [..., cols/32], raw E8M0 (value scale = 2^(u8-127)).
  OCP MXFP4 element values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

Bit-exactness: ggml computes (2*v) * (2^(e-127)/2), HF computes v * 2^(e-127). Both are
exact power-of-two rescalings of the same value -> identical float32 bit patterns
(including the e<2 subnormal-scale cases, handled explicitly below).
"""
from __future__ import annotations

import numpy as np

QK = 32  # values per block (ggml QK_MXFP4)
BLOCK_BYTES = 1 + QK // 2  # e + qs[16]

# ggml kvalues_mxfp4 (doubled E2M1)
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                   dtype=np.float32)
# OCP E2M1 element values (HF-side semantics)
E2M1 = KVALUES * np.float32(0.5)


def e8m0_to_fp32(e: np.ndarray) -> np.ndarray:
    """OCP E8M0 scale: 2^(e-127); e==0 -> 2^-127 (mirrors ggml_e8m0_to_fp32)."""
    e = e.astype(np.uint32)
    bits = np.where(e == 0, np.uint32(0x00400000), e << np.uint32(23))
    return bits.view(np.float32)


def e8m0_to_fp32_half(e: np.ndarray) -> np.ndarray:
    """ggml_e8m0_to_fp32_half (ggml-impl.h:477-495): 2^(e-127)/2 incl. e<2 denormals."""
    e = e.astype(np.uint32)
    bits = np.where(e < 2,
                    np.uint32(0x00200000) << e,
                    (e - np.uint32(1)) << np.uint32(23))
    return bits.view(np.float32)


def _split_nibbles_hf(blocks: np.ndarray) -> np.ndarray:
    """uint8 [..., cols/2] -> logical values uint8 [..., cols] (adjacent low/high)."""
    low = blocks & 0x0F
    high = blocks >> 4
    return np.stack([low, high], axis=-1).reshape(*blocks.shape[:-1], -1)


def dequant_hf(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """HF/OCP reference dequant. blocks [..., cols/2] u8, scales [..., cols/32] u8."""
    vals = E2M1[_split_nibbles_hf(blocks)]                       # [..., cols]
    d = e8m0_to_fp32(scales)                                     # [..., cols/32]
    with np.errstate(over="ignore"):  # e≥253 × max mantissa exceeds f32 → inf on BOTH sides
        return (vals.reshape(*d.shape, QK) * d[..., None]).reshape(vals.shape)


def repack_hf_to_ggml(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """[..., cols/2] u8 + [..., cols/32] u8 -> ggml raw bytes [..., cols/32 * 17].

    Per block: byte 0 = e (E8M0, unchanged); qs[j] = nibble(value j) | nibble(value j+16) << 4.
    """
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise TypeError("blocks/scales must be uint8")
    if np.any(scales == 0xFF):
        raise ValueError("E8M0 scale byte 0xFF (NaN) in source input — ggml does not "
                         "handle it (ggml-impl.h: 'NaNs are not handled here'); refusing "
                         "to repack")
    cols = blocks.shape[-1] * 2
    if cols % QK:
        raise ValueError(f"row has {cols} values, not a multiple of {QK}")
    n_blocks = cols // QK
    if scales.shape != (*blocks.shape[:-1], n_blocks):
        raise ValueError(f"scales shape {scales.shape} != {(*blocks.shape[:-1], n_blocks)}")

    vals = _split_nibbles_hf(blocks).reshape(*blocks.shape[:-1], n_blocks, QK)
    qs = vals[..., :16] | (vals[..., 16:] << 4)                  # [..., n_blocks, 16]
    raw = np.concatenate([scales[..., None], qs], axis=-1)       # [..., n_blocks, 17]
    return np.ascontiguousarray(raw.reshape(*blocks.shape[:-1], n_blocks * BLOCK_BYTES))


def dequant_ggml(raw: np.ndarray) -> np.ndarray:
    """Mirror of dequantize_row_mxfp4 (ggml-quants.c) over ggml raw bytes [..., nb*17]."""
    nb = raw.shape[-1] // BLOCK_BYTES
    b = raw.reshape(*raw.shape[:-1], nb, BLOCK_BYTES)
    d = e8m0_to_fp32_half(b[..., 0])                             # [..., nb]
    qs = b[..., 1:]
    lo = KVALUES[qs & 0x0F]                                      # values 0..15
    hi = KVALUES[qs >> 4]                                        # values 16..31
    with np.errstate(over="ignore"):  # matches dequant_hf: shared inf edge at e≥253
        out = np.concatenate([lo, hi], axis=-1) * d[..., None]   # [..., nb, 32]
    return out.reshape(*raw.shape[:-1], nb * QK)


def unpack_ggml_to_hf(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of repack_hf_to_ggml (for round-trip testing)."""
    nb = raw.shape[-1] // BLOCK_BYTES
    b = raw.reshape(*raw.shape[:-1], nb, BLOCK_BYTES)
    scales = np.ascontiguousarray(b[..., 0])
    qs = b[..., 1:]
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)         # [..., nb, 32] logical order
    pairs = vals.reshape(*vals.shape[:-2], nb * QK).reshape(*vals.shape[:-2], -1, 2)
    blocks = (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)
    return np.ascontiguousarray(blocks), scales
