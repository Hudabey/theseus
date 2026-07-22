# Reference sources

The recon docs cite `vendor/<repo>/<path>:<line>` throughout. The `vendor/` clones are
not distributed with this repo; to make every file:line citation resolve, clone these
repos at exactly these commits:

| cited as | repo | commit inspected |
|---|---|---|
| `vendor/llama.cpp` | https://github.com/ggml-org/llama.cpp | `1a064ab0921238c1daa397d6f4a900ef33884de2` (2026-07-22) |
| `vendor/ik_llama.cpp` | https://github.com/ikawrakow/ik_llama.cpp | `e5357286c0d433cd4384e82ed7e2b6d655f57087` (2026-07-22) |
| `vendor/fla` | https://github.com/fla-org/flash-linear-attention | `d1ce07369d581813553f30a750af3b6b5f9af6a9` (2026-07-22) |
| `vendor/kimi-linear` | https://github.com/MoonshotAI/Kimi-Linear | `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` (2025-11-17) |

```bash
mkdir -p vendor && cd vendor
git clone https://github.com/ggml-org/llama.cpp          && git -C llama.cpp     checkout 1a064ab0921238c1daa397d6f4a900ef33884de2
git clone https://github.com/ikawrakow/ik_llama.cpp      && git -C ik_llama.cpp  checkout e5357286c0d433cd4384e82ed7e2b6d655f57087
git clone https://github.com/fla-org/flash-linear-attention fla && git -C fla    checkout d1ce07369d581813553f30a750af3b6b5f9af6a9
git clone https://github.com/MoonshotAI/Kimi-Linear kimi-linear && git -C kimi-linear checkout 8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3
```

Additional cited sources:

- Kimi-Linear tech report — `vendor/kimi-linear/tech_report.pdf` (in the repo above).
- Attention Residuals paper — arXiv 2603.15031 (cited by `fla/ops/attnres/naive.py:31`).
- vLLM K3 preview post — https://vllm.ai/blog/2026-07-22-kimi-k3-preview (fetched
  2026-07-22; quoted verbatim in recon/05).
