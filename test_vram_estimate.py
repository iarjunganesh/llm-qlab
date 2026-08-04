"""Tests for the VRAM headroom estimate.

The bug these exist to prevent: the guard multiplied GGUF file size by a flat
1.15. That cannot express a KV cache which, across the three families
benchmarked here, spans 28 MB to 256 MB at 512 tokens on nearly identical file
sizes — a 9x range driven entirely by grouped-query attention. It also charged
for weights that never reach the device: Qwen2.5-7B keeps ~550 MB of embedding
table on the host even at full offload.
"""

import pytest

from bench_core import (
    UNKNOWN,
    _meta_int,
    bytes_per_weight,
    estimate_host_embedding_mb,
    estimate_kv_cache_mb,
)

# Geometry as actually read from each GGUF header.
LLAMA2 = {"layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 128,
          "embedding": 4096, "vocab_size": 32000}
MISTRAL = {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128,
           "embedding": 4096, "vocab_size": 32000}
QWEN25 = {"layers": 28, "heads": 28, "kv_heads": 4, "head_dim": 128,
          "embedding": 3584, "vocab_size": 152064}


class TestHostEmbeddingEstimate:
    """Checked against host residency llama.cpp actually reported.

    llama.cpp leaves token_embd on the CPU at full offload, so GGUF file size
    overstates VRAM demand. For Qwen2.5 that gap is 552 MB — enough to refuse a
    model that fits with 278 MB to spare.
    """

    @pytest.mark.parametrize("geometry,quant,observed_mb", [
        (LLAMA2, "Q4_K_M", 70),
        (LLAMA2, "Q5_K_M", 86),
        (LLAMA2, "Q8_0", 133),
        (MISTRAL, "Q8_0", 133),
        (QWEN25, "Q8_0", 552),
    ])
    def test_matches_measured_host_residency(self, geometry, quant, observed_mb):
        estimate = estimate_host_embedding_mb(geometry, quant)
        assert estimate == pytest.approx(observed_mb, rel=0.02)

    def test_qwen_table_dwarfs_llama_at_similar_file_size(self):
        """The 152k vocabulary is what breaks a file-size heuristic."""
        gap = (estimate_host_embedding_mb(QWEN25, "Q8_0")
               - estimate_host_embedding_mb(LLAMA2, "Q8_0"))
        assert gap > 400

    def test_unknown_quant_yields_unknown_not_zero(self):
        """A zero would understate demand and admit a model that cannot fit."""
        assert estimate_host_embedding_mb(LLAMA2, "Q3_K_XL_MADE_UP") == UNKNOWN
        assert estimate_host_embedding_mb(LLAMA2, "") == UNKNOWN

    def test_missing_vocab_yields_unknown(self):
        assert estimate_host_embedding_mb({"embedding": 4096}, "Q8_0") == UNKNOWN


class TestBytesPerWeight:
    @pytest.mark.parametrize("label,expected", [
        ("Q8_0", 34 / 32),
        ("Q4_K_M", 144 / 256),
        ("Q4_K_S", 144 / 256),
        ("Q5_K_M", 176 / 256),
        ("Q6_K", 210 / 256),
    ])
    def test_size_suffix_is_trimmed_to_the_block_format(self, label, expected):
        assert bytes_per_weight(label) == pytest.approx(expected)

    def test_case_and_whitespace_tolerant(self):
        assert bytes_per_weight("  q8_0 ") == pytest.approx(34 / 32)

    def test_unrecognized_label_is_unknown(self):
        assert bytes_per_weight("NOT_A_QUANT") == UNKNOWN


class TestKvCacheEstimate:
    @pytest.mark.parametrize("geometry,expected_mb", [
        (LLAMA2, 256.0),   # full MHA: 32 KV heads
        (MISTRAL, 64.0),   # GQA 8
        (QWEN25, 28.0),    # GQA 4, and only 28 layers
    ])
    def test_matches_hand_computed_size_at_512_ctx(self, geometry, expected_mb):
        assert estimate_kv_cache_mb(geometry, 512) == pytest.approx(expected_mb, rel=1e-3)

    def test_the_flat_multiplier_could_not_have_expressed_this(self):
        """The spread a percentage-of-file-size heuristic has to miss."""
        biggest = estimate_kv_cache_mb(LLAMA2, 512)
        smallest = estimate_kv_cache_mb(QWEN25, 512)
        assert biggest / smallest > 9

    def test_scales_linearly_with_context(self):
        assert estimate_kv_cache_mb(LLAMA2, 1024) == pytest.approx(
            2 * estimate_kv_cache_mb(LLAMA2, 512))

    def test_incomplete_geometry_is_unknown_not_guessed(self):
        assert estimate_kv_cache_mb({"layers": 32}, 512) == UNKNOWN
        assert estimate_kv_cache_mb({}, 512) == UNKNOWN

    def test_zero_values_do_not_produce_a_zero_estimate(self):
        """A zero would read as 'no KV cache' and wrongly admit the model."""
        assert estimate_kv_cache_mb(
            {"layers": 0, "kv_heads": 32, "head_dim": 128}, 512) == UNKNOWN


class TestMetadataLookup:
    """GGUF keys are namespaced by architecture, not by a fixed prefix."""

    def test_finds_key_regardless_of_architecture_prefix(self):
        assert _meta_int({"llama.block_count": "32"}, "block_count") == 32
        assert _meta_int({"qwen2.block_count": "28"}, "block_count") == 28

    def test_dotted_suffix_does_not_match_a_partial_key(self):
        meta = {"llama.attention.head_count": "32",
                "llama.attention.head_count_kv": "8"}
        assert _meta_int(meta, "attention.head_count_kv") == 8

    def test_missing_key_is_negative_not_zero(self):
        assert _meta_int({"llama.block_count": "32"}, "embedding_length") == -1

    def test_unparseable_value_is_negative(self):
        assert _meta_int({"llama.block_count": "many"}, "block_count") == -1


class TestHeadDimDerivation:
    """key_length is optional in GGUF and absent from all three families here."""

    @pytest.mark.parametrize("geometry", [LLAMA2, MISTRAL, QWEN25])
    def test_embedding_over_heads_gives_the_real_head_dim(self, geometry):
        assert geometry["embedding"] // geometry["heads"] == geometry["head_dim"]
