# Converting Kimi K3 to GGUF with theseus

`theseus convert` turns the released `moonshotai/Kimi-K3` checkpoint
(~1.55 TB, 96 shards) into GGUF with the routed experts' native MXFP4
**preserved byte-exactly** — a pure byte re-pairing into ggml's
`block_mxfp4`, never a requantization. The BF16 skeleton (attention, shared
experts, latent sandwich, router, norms, embeddings) is written at source
precision. Anyone can re-verify the passthrough claim: the repack is
involutive (`theseus/repack_mxfp4.py` ships both directions plus both
dequant references, oracle-tested).

## Machine

CPU-only. No GPU is involved in conversion.

- **disk:** ≥ 3.6 TB NVMe (1.55 TB input + ~1.55 TB output + headroom)
- **RAM:** 64 GB is comfortable (the converter streams; peak is one layer's
  expert stack, ~6 GB)
- **network:** the fatter the better — the wall clock is dominated by
  download/upload, not compute

## Steps

```bash
# 0. environment
python3 -m venv venv && . venv/bin/activate
git clone https://github.com/Hudabey/theseus && cd theseus
pip install -e '.[convert]'
pip install 'huggingface_hub[cli,hf_transfer]'

# 1. download the checkpoint (resumable; ~1.55 TB)
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download moonshotai/Kimi-K3 --local-dir /data/kimi-k3

# 2. convert (--dry-run first: plans the full tensor map in seconds,
#    catches config/name surprises before any bytes move)
theseus convert /data/kimi-k3 --outfile /data/kimi-k3-gguf --dry-run
theseus convert /data/kimi-k3 --outfile /data/kimi-k3-gguf --split-max-size 45G

# 3. verify + manifest
sha256sum /data/kimi-k3-gguf/*.gguf > /data/kimi-k3-gguf/SHA256SUMS
```

The converter hard-refuses any checkpoint whose config does not match the
audited K3 release values (shape, layer schedule, quantization format) —
`--allow-nonrelease-shapes` exists only for the test fixtures.

## What lands in the GGUF

- arch `kimi-k3`; per-layer `head_count_kv` (0 = KDA, 1 = MLA — 69/24,
  final layer full-attention)
- MXFP4 routed experts, byte-identical to Moonshot's release (spot-check:
  unpack any expert with `theseus.repack_mxfp4.unpack_ggml_to_hf` and
  compare against the safetensors shard)
- AttnRes tensors (per-sublayer + model-level) and the Stable LatentMoE
  sandwich, BF16
- the K3 metadata contract (`theseus/_vendor/PATCHES.md`): KDA gate lower
  bound, AttnRes block size, SiTU betas, latent dim
- generation EOS `<|end_of_msg|>` (163586) — the tokenizer's `[EOS]`
  (163585) is *not* the stop token; converters that copy it produce
  models that never stop in chat

## Notes

- The emitted GGUF targets the `kimi-k3` architecture; runtime support
  ships on theseus's clock (the GGUF is the stable contract).
- Tested end-to-end on a synthetic K3-shaped checkpoint
  (`tests/test_convert_k3_e2e.py`) including byte-exact MXFP4 assertions,
  and the tokenizer path is verified against the released tokenizer files
  (pre-tokenizer resolves to `kimi-k2`; 163,840/163,840 ids covered).
