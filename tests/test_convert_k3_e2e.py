"""End-to-end test of `theseus convert` on a synthetic Kimi-K3-shaped checkpoint.

The fixture mirrors the released moonshotai/Kimi-K3 structure exactly — VL
wrapper prefix, compressed-tensors MXFP4 packed/scale U8 expert pairs, latent
sandwich, AttnRes tensors, full-rank output gates on every layer, hybrid
KDA/MLA schedule with the final layer full-attention — with dimensions shrunk
(all divisible by the MXFP4 group of 32 where quantized). The converter runs
the same code path the real 1.55 TB conversion runs; the emitted GGUF is then
torn apart with gguf_reader and every claim checked, including byte-exact
MXFP4 passthrough against the oracle repack.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theseus import convert_k3  # noqa: E402
from theseus.repack_mxfp4 import repack_hf_to_ggml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theseus" / "_vendor"))
import gguf  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

# --- fixture dims (K3-faithful, shrunk) --------------------------------------
H = 64            # hidden_size          (release: 7168)
LAYERS = 5        # num_hidden_layers    (release: 93)
HEADS = 4         # num_attention_heads  (release: 96)
KDA_DIM = 32      # heads * kda head_dim (release: 12288); == heads*v_head_dim
KDA_HEAD = 8      # kda head_dim         (release: 128)
LATENT = 64       # routed_expert_hidden (release: 3584)
MOE_INT = 32      # moe_intermediate     (release: 3072)
E = 8             # num_experts          (release: 896)
K = 2             # experts per token    (release: 16)
FULL_ATTN = [4, 5]  # 1-based MLA layers (release: {4,8,...,92} ∪ {93})
VOCAB = 128
Q_LORA, KV_LORA, NOPE, ROPE, V_HEAD = 16, 16, 8, 4, 8
DENSE_INT = 128
SHARED_INT = 2 * MOE_INT

P = "language_model."


def _bf16(*shape):
    return (torch.randn(*shape) * 0.02).to(torch.bfloat16)


def build_checkpoint(dir_model: Path):
    """Returns {(bid, xid, wid): (blocks, scales)} for expert byte assertions."""
    t: dict[str, torch.Tensor] = {}
    expert_src = {}

    t[P + "model.embed_tokens.weight"] = _bf16(VOCAB, H)
    t[P + "model.norm.weight"] = _bf16(H)
    t[P + "model.output_attn_res_proj.weight"] = _bf16(1, H)
    t[P + "model.output_attn_res_norm.weight"] = _bf16(H)
    t[P + "lm_head.weight"] = _bf16(VOCAB, H)
    # must be skipped by the converter:
    t["vision_tower.blocks.0.attn.qkv.weight"] = _bf16(8, 8)
    t["mm_projector.proj.weight"] = _bf16(8, 8)

    for i in range(LAYERS):
        L = f"{P}model.layers.{i}."
        t[L + "input_layernorm.weight"] = _bf16(H)
        t[L + "post_attention_layernorm.weight"] = _bf16(H)
        t[L + "self_attention_res_proj.weight"] = _bf16(1, H)
        t[L + "self_attention_res_norm.weight"] = _bf16(H)
        t[L + "mlp_res_proj.weight"] = _bf16(1, H)
        t[L + "mlp_res_norm.weight"] = _bf16(H)
        t[L + "self_attn.g_proj.weight"] = _bf16(KDA_DIM, H)
        t[L + "self_attn.o_proj.weight"] = _bf16(H, KDA_DIM)

        if (i + 1) in FULL_ATTN:  # MLA
            t[L + "self_attn.q_a_proj.weight"] = _bf16(Q_LORA, H)
            t[L + "self_attn.q_a_layernorm.weight"] = _bf16(Q_LORA)
            t[L + "self_attn.q_b_proj.weight"] = _bf16(HEADS * (NOPE + ROPE), Q_LORA)
            t[L + "self_attn.kv_a_proj_with_mqa.weight"] = _bf16(KV_LORA + ROPE, H)
            t[L + "self_attn.kv_a_layernorm.weight"] = _bf16(KV_LORA)
            t[L + "self_attn.kv_b_proj.weight"] = _bf16(HEADS * (NOPE + V_HEAD), KV_LORA)
        else:  # KDA
            for p in ("q", "k", "v"):
                t[L + f"self_attn.{p}_proj.weight"] = _bf16(KDA_DIM, H)
                t[L + f"self_attn.{p}_conv1d.weight"] = _bf16(KDA_DIM, 1, 4)
            t[L + "self_attn.f_a_proj.weight"] = _bf16(KDA_HEAD, H)
            t[L + "self_attn.f_b_proj.weight"] = _bf16(KDA_DIM, KDA_HEAD)
            t[L + "self_attn.b_proj.weight"] = _bf16(HEADS, H)
            t[L + "self_attn.A_log"] = torch.rand(HEADS).to(torch.bfloat16) + 1.0
            t[L + "self_attn.dt_bias"] = _bf16(KDA_DIM)
            t[L + "self_attn.o_norm.weight"] = _bf16(KDA_HEAD)

        if i == 0:  # dense MLP
            t[L + "mlp.gate_proj.weight"] = _bf16(DENSE_INT, H)
            t[L + "mlp.up_proj.weight"] = _bf16(DENSE_INT, H)
            t[L + "mlp.down_proj.weight"] = _bf16(H, DENSE_INT)
        else:  # MoE
            B = L + "block_sparse_moe."
            t[B + "gate.weight"] = _bf16(E, H)
            t[B + "gate.e_score_correction_bias"] = torch.randn(E).float()
            t[B + "routed_expert_down_proj.weight"] = _bf16(LATENT, H)
            t[B + "routed_expert_norm.weight"] = _bf16(LATENT)
            t[B + "routed_expert_up_proj.weight"] = _bf16(H, LATENT)
            t[B + "shared_experts.gate_proj.weight"] = _bf16(SHARED_INT, H)
            t[B + "shared_experts.up_proj.weight"] = _bf16(SHARED_INT, H)
            t[B + "shared_experts.down_proj.weight"] = _bf16(H, SHARED_INT)
            for x in range(E):
                for wid, (rows, cols) in (("w1", (MOE_INT, LATENT)),
                                          ("w2", (LATENT, MOE_INT)),
                                          ("w3", (MOE_INT, LATENT))):
                    blocks = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
                    scales = torch.randint(100, 140, (rows, cols // 32), dtype=torch.uint8)
                    t[f"{B}experts.{x}.{wid}.weight_packed"] = blocks
                    t[f"{B}experts.{x}.{wid}.weight_scale"] = scales
                    expert_src[(i, x, wid)] = (blocks.numpy(), scales.numpy())

    shard = "model-00001-of-00001.safetensors"
    save_file(t, str(dir_model / shard))
    index = {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in t.values())},
             "weight_map": {k: shard for k in t}}
    (dir_model / "model.safetensors.index.json").write_text(json.dumps(index))

    config = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "tie_word_embeddings": False,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 4, "type": "float", "symmetric": True,
                "strategy": "group", "group_size": 32}}},
        },
        "text_config": {
            "model_type": "kimi_linear",
            "hidden_size": H,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": HEADS,
            "intermediate_size": DENSE_INT,
            "num_experts": E,
            "num_experts_per_token": K,
            "num_shared_experts": 2,
            "moe_intermediate_size": MOE_INT,
            "routed_expert_hidden_size": LATENT,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "vocab_size": VOCAB,
            "max_position_embeddings": 512,
            "attn_res_block_size": 2,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "rms_norm_eps": 1e-5,
            "q_lora_rank": Q_LORA,
            "kv_lora_rank": KV_LORA,
            "qk_nope_head_dim": NOPE,
            "qk_rope_head_dim": ROPE,
            "v_head_dim": V_HEAD,
            "linear_attn_config": {
                "head_dim": KDA_HEAD,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "full_attn_layers": FULL_ATTN,
                "kda_layers": [x for x in range(1, LAYERS + 1) if x not in FULL_ATTN],
            },
        },
        "vision_config": {"model_type": "kimi_k3_vision"},
    }
    (dir_model / "config.json").write_text(json.dumps(config, indent=1))
    return expert_src


class _FixtureModel(convert_k3.KimiK3Model):
    """Real converter with only the tokenizer stubbed (tokenizer has its own
    verification against the actual K3 tokenizer files)."""
    model_arch = gguf.MODEL_ARCH.KIMI_K3

    def set_vocab(self):
        # byte-level stub: ids 0..127 are the gpt2 byte symbols for bytes
        # 0..127, so any ASCII prompt tokenizes char-by-char (needed by the
        # runtime fixture harness); no merges
        from theseus._vendor.conversion.qwen import QwenModel
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("kimi-k2")
        self.gguf_writer.add_token_list(
            [QwenModel.token_bytes_to_string(bytes([i])) for i in range(VOCAB)])
        self.gguf_writer.add_token_types([int(gguf.TokenType.NORMAL)] * VOCAB)
        self.gguf_writer.add_token_merges([])
        self.gguf_writer.add_eos_token_id(2)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    src = tmp_path_factory.mktemp("k3-mini")
    out = tmp_path_factory.mktemp("k3-mini-gguf")
    expert_src = build_checkpoint(src)
    model = _FixtureModel(src, gguf.LlamaFileType.MOSTLY_BF16, out,
                          split_max_tensors=0, split_max_size=0, dry_run=False)
    model.expect_release = False
    model.write()
    files = sorted(out.glob("*.gguf"))
    assert len(files) == 1, files
    return gguf.GGUFReader(files[0]), expert_src


def _kv(reader, key):
    field = reader.get_field(key)
    assert field is not None, f"missing KV {key!r}"
    return field.contents()


def test_arch_and_kv_contract(converted):
    r, _ = converted
    assert _kv(r, "general.architecture") == "kimi-k3"
    assert _kv(r, "kimi-k3.block_count") == LAYERS
    assert list(_kv(r, "kimi-k3.attention.head_count_kv")) == [0, 0, 0, 1, 1]
    assert _kv(r, "kimi-k3.ssm.conv_kernel") == 4
    assert _kv(r, "kimi-k3.kda.head_dim") == KDA_HEAD
    assert _kv(r, "kimi-k3.kda.gate_lower_bound") == pytest.approx(-5.0)
    assert _kv(r, "kimi-k3.attnres.block_size") == 2
    assert _kv(r, "kimi-k3.situ.beta") == pytest.approx(4.0)
    assert _kv(r, "kimi-k3.situ.linear_beta") == pytest.approx(25.0)
    assert _kv(r, "kimi-k3.latent_moe.dim") == LATENT
    assert _kv(r, "kimi-k3.expert_count") == E
    assert _kv(r, "kimi-k3.expert_used_count") == K
    assert _kv(r, "kimi-k3.expert_shared_count") == 2
    assert _kv(r, "kimi-k3.leading_dense_block_count") == 1
    assert _kv(r, "kimi-k3.expert_feed_forward_length") == MOE_INT
    assert _kv(r, "general.file_type") == int(gguf.LlamaFileType.MOSTLY_MXFP4_MOE)


def test_tensor_inventory(converted):
    r, _ = converted
    names = {t.name for t in r.tensors}

    assert not any("vision" in n or "mm_projector" in n for n in names)

    for i in range(LAYERS):
        assert f"blk.{i}.attn_gate.weight" in names          # output gate, all layers
        for base in ("attn_res_proj", "attn_res_norm", "ffn_res_proj", "ffn_res_norm"):
            assert f"blk.{i}.{base}.weight" in names
    assert "res_proj.weight" in names and "res_norm.weight" in names

    for i in (0, 1, 2):  # KDA layers
        for base in ("ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v",
                     "ssm_f_a", "ssm_f_b", "ssm_beta", "ssm_norm"):
            assert f"blk.{i}.{base}.weight" in names, f"blk.{i}.{base}"
        assert f"blk.{i}.ssm_a" in names
        assert f"blk.{i}.ssm_dt.bias" in names
    for i in (3, 4):     # MLA layers: split kv_b present
        assert f"blk.{i}.attn_k_b.weight" in names
        assert f"blk.{i}.attn_v_b.weight" in names

    assert "blk.0.ffn_gate.weight" in names                  # dense layer 0
    for i in (1, 2, 3, 4):                                   # MoE layers
        for base in ("ffn_latent_down", "ffn_latent_up", "ffn_latent_norm",
                     "ffn_gate_inp", "ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp"):
            assert f"blk.{i}.{base}.weight" in names, f"blk.{i}.{base}"
        assert f"blk.{i}.exp_probs_b.bias" in names


def test_mxfp4_passthrough_byte_exact(converted):
    r, expert_src = converted
    by_name = {t.name: t for t in r.tensors}
    wid_to_tensor = {"w1": gguf.MODEL_TENSOR.FFN_GATE_EXP,
                     "w2": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
                     "w3": gguf.MODEL_TENSOR.FFN_UP_EXP}
    for i in (1, 2, 3, 4):
        for wid, key in wid_to_tensor.items():
            name = gguf.TENSOR_NAMES[key].format(bid=i) + ".weight"
            t = by_name[name]
            assert t.tensor_type == gguf.GGMLQuantizationType.MXFP4, name
            expected = np.stack(
                [repack_hf_to_ggml(*expert_src[(i, x, wid)]) for x in range(E)]
            )
            got = np.asarray(t.data).reshape(expected.shape)
            assert np.array_equal(got, expected), f"MXFP4 bytes differ for {name}"


def test_a_log_baked(converted):
    r, _ = converted
    by_name = {t.name: t for t in r.tensors}
    t = by_name["blk.0.ssm_a"]
    data = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        vals = data.view(np.float32)
    else:
        vals = data.astype(np.float32)
    # source A_log was in [1, 2]; -exp of that is in [-e^2, -e]
    assert (vals < 0).all(), "A_log must be stored as -exp(A_log)"
    assert (vals > -8.0).all() and (vals < -2.5).all()
