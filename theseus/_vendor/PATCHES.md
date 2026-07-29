# Vendored conversion machinery

`gguf/` and `conversion/` are vendored from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) @
`1a064ab0921238c1daa397d6f4a900ef33884de2` (MIT, see `LICENSE.llama.cpp`):
`gguf-py/gguf/` → `gguf/` (scripts dropped), `conversion/` → `conversion/`.

theseus modifications on top of upstream — Kimi K3 architecture support
(`kimi-k3` does not exist upstream yet):

## `gguf/constants.py`

- `MODEL_ARCH.KIMI_K3` + name `"kimi-k3"`.
- `MODEL_TENSOR` additions with GGUF names:
  - AttnRes (per-layer): `ATTN_RES_PROJ` `blk.{bid}.attn_res_proj`,
    `ATTN_RES_NORM` `blk.{bid}.attn_res_norm`, `FFN_RES_PROJ`
    `blk.{bid}.ffn_res_proj`, `FFN_RES_NORM` `blk.{bid}.ffn_res_norm`
  - AttnRes (model-level): `RES_PROJ` `res_proj`, `RES_NORM` `res_norm`
  - Stable LatentMoE sandwich: `FFN_LATENT_DOWN` `blk.{bid}.ffn_latent_down`,
    `FFN_LATENT_UP` `blk.{bid}.ffn_latent_up`, `FFN_LATENT_NORM`
    `blk.{bid}.ffn_latent_norm`
- `MODEL_TENSORS[MODEL_ARCH.KIMI_K3]`: KIMI_LINEAR list + `ATTN_GATE`
  (full-rank output gate, present on ALL 93 layers — KDA and MLA share the
  slot, disambiguated by layer type) + the nine tensors above.
- New KV keys (the metadata contract the K3 runtime reads):
  - `{arch}.kda.gate_lower_bound` (f32) — safe-gate bound, −5.0 in the release
  - `{arch}.attnres.block_size` (u32) — 12 in the release
  - `{arch}.situ.beta` (f32) / `{arch}.situ.linear_beta` (f32) — 4.0 / 25.0
  - `{arch}.latent_moe.dim` (u32) — 3584 in the release

## `gguf/gguf_writer.py`

- Writer methods for the five new KVs: `add_kda_gate_lower_bound`,
  `add_attnres_block_size`, `add_situ_beta`, `add_situ_linear_beta`,
  `add_latent_moe_dim`.

## `conversion/qwen.py`

- `token_bytes_to_string`: `bytes_to_unicode` import falls back to
  `transformers.convert_slow_tokenizer` — the symbol moved there in
  transformers 5.x and the upstream import path only exists in ≤4.x.

Every value is asserted against the checkpoint's `config.json` at conversion
time; none is trusted from documentation.
