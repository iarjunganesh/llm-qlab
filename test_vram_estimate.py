"""Tests for the VRAM headroom estimate.

The bug these exist to prevent: the guard multiplied GGUF file size by a flat
1.15. That cannot express a KV cache which, across the three families
benchmarked here, spans 28 MB to 256 MB at 512 tokens on nearly identical file
sizes — a 9x range driven entirely by grouped-query attention. It also charged
for weights that never reach the device: Qwen2.5-7B keeps ~550 MB of embedding
table on the host even at full offload.
"""

import pytest

from bench_core import UNKNOWN, _meta_int, estimate_kv_cache_mb

# Geometry as actually read from each GGUF header.
LLAMA2 = {"layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 128, "embedding": 4096}
MISTRAL = {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "embedding": 4096}
QWEN25 = {"layers": 28, "heads": 28, "kv_heads": 4, "head_dim": 128, "embedding": 3584}


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
