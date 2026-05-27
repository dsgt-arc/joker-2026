"""
Contract tests for the corrected retrieval bucketization architecture.

These tests are intentionally architecture-level tests. They are expected to fail
until the production retrieval module implements bucketization as four explicit
retrieval regimes:

    A0_B0 = phonetic_matches(A0, B0)
    A1_B0 = phonetic_matches(A1, B0)
    B1_A0 = phonetic_matches(B1, A0)
    A1_B1 = phonetic_matches(A1, B1)

Where:
    A0 = original first_meaning_fr list items with IPA
    B0 = original second_meaning_fr list items with IPA
    A1 = semantic-near-A candidates with IPA, including FastText candidates
    B1 = semantic-near-B candidates with IPA, including FastText candidates

The tests require explicit bucket-regime diagnostics so a single merged A* x B*
matrix with post-hoc labels cannot satisfy the contract silently.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


EXPECTED_BUCKET_IDS = ["A0_B0", "A1_B0", "B1_A0", "A1_B1"]

EXPECTED_BUCKET_REGIMES = [
    {"bucket_id": "A0_B0", "source_pool": "A0", "target_pool": "B0"},
    {"bucket_id": "A1_B0", "source_pool": "A1", "target_pool": "B0"},
    {"bucket_id": "B1_A0", "source_pool": "B1", "target_pool": "A0"},
    {"bucket_id": "A1_B1", "source_pool": "A1", "target_pool": "B1"},
]

EXPECTED_FINAL_BUCKET_COUNTS = {"A0_B0": 1, "A1_B0": 1, "B1_A0": 1, "A1_B1": 1}


def _install_import_stubs() -> None:
    """Stub heavyweight project/model imports so the module can be contract-tested."""

    if "faiss" not in sys.modules:
        faiss = types.ModuleType("faiss")
        faiss.read_index = lambda *args, **kwargs: None
        sys.modules["faiss"] = faiss

    if "sentence_transformers" not in sys.modules:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, *args, **kwargs):
                texts = list(texts)
                if not texts:
                    return np.zeros((0, 4), dtype="float32")
                return np.ones((len(texts), 4), dtype="float32")

        sentence_transformers.SentenceTransformer = DummySentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        sys.modules["sklearn"] = sklearn

    if "sklearn.feature_extraction" not in sys.modules:
        sys.modules["sklearn.feature_extraction"] = types.ModuleType("sklearn.feature_extraction")

    if "sklearn.feature_extraction.text" not in sys.modules:
        text_mod = types.ModuleType("sklearn.feature_extraction.text")

        class DummyTfidfVectorizer:
            def __init__(self, *args, **kwargs):
                pass

            def fit_transform(self, docs):
                return np.zeros((len(list(docs)), 1), dtype="float32")

            def transform(self, docs):
                return np.zeros((len(list(docs)), 1), dtype="float32")

        text_mod.TfidfVectorizer = DummyTfidfVectorizer
        sys.modules["sklearn.feature_extraction.text"] = text_mod

    if "sklearn.metrics" not in sys.modules:
        sys.modules["sklearn.metrics"] = types.ModuleType("sklearn.metrics")

    if "sklearn.metrics.pairwise" not in sys.modules:
        pairwise = types.ModuleType("sklearn.metrics.pairwise")
        pairwise.linear_kernel = lambda a, b: np.zeros((len(a), len(b)), dtype="float32")
        sys.modules["sklearn.metrics.pairwise"] = pairwise

    if "data" not in sys.modules:
        data = types.ModuleType("data")
        data.load = lambda *args, **kwargs: pd.DataFrame()
        data.load_all = lambda *args, **kwargs: pd.DataFrame()
        data.save = lambda df, path: df.to_csv(path, sep="\t", index=False)
        sys.modules["data"] = data

    if "config" not in sys.modules:
        config = types.ModuleType("config")
        config.translate_dir = ""
        config.phonetic_items_path = ""
        config.phonetic_index_path = ""
        config.phonetic_model_path = ""
        config.fasttext_fr_path = ""
        config.phonetic_embeddings_path = ""
        sys.modules["config"] = config


def load_retrieval_module():
    """Load the retrieval module named by RETRIEVAL_MODULE_UNDER_TEST."""
    _install_import_stubs()
    module_path = Path(os.environ.get("RETRIEVAL_MODULE_UNDER_TEST", "retrieval_bucketized_french_only.py"))
    if not module_path.exists():
        pytest.fail(
            "Set RETRIEVAL_MODULE_UNDER_TEST to the retrieval file under test, "
            f"or place retrieval_bucketized_french_only.py in the test working directory. Got: {module_path}"
        )

    spec = importlib.util.spec_from_file_location("retrieval_under_bucketization_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["retrieval_under_bucketization_test"] = module
    spec.loader.exec_module(module)
    return module


class FakeExpression:
    """Only the methods touched by retrieve_row/mine_bridges are implemented."""

    def semantic_search_many(self, requests):
        out: dict[str, list[dict[str, Any]]] = {}
        for query, _top_k, channel in requests:
            if channel.startswith("semantic_A_term:"):
                out[channel] = [
                    {"surface": "semalpha", "score": 0.95, "source": "semantic_fixture"},
                    # Duplicate original seed rediscovered semantically. This
                    # can appear in raw A1 diagnostics, but must not take final
                    # ownership away from A0_B0.
                    {"surface": "alpha", "score": 0.90, "source": "semantic_fixture_duplicate"},
                ]
            elif channel.startswith("semantic_B_term:"):
                out[channel] = [
                    {"surface": "sembeta", "score": 0.95, "source": "semantic_fixture"},
                    # Duplicate original seed rediscovered semantically.
                    {"surface": "beta", "score": 0.90, "source": "semantic_fixture_duplicate"},
                ]
            elif channel == "semantic_blended":
                out[channel] = []
            else:
                out[channel] = []
        return out

    def lexical_search(self, *args, **kwargs):
        return []

    def semantic_scores(self, query, texts, batch_size=64):
        return [0.75 for _ in texts]


class FakeFastText:
    enabled = True

    def __init__(self):
        self.last_stats = {
            "fasttext_enabled": True,
            "fasttext_seed_count": 0,
            "fasttext_expansion_count": 0,
            "fasttext_budget_filled": False,
        }

    def expand(self, seeds, side, level=1, limit=None):
        if side == "A":
            rows = [{"surface": "fastalpha", "text": "fastalpha"}]
        elif side == "B":
            rows = [{"surface": "fastbeta", "text": "fastbeta"}]
        else:
            rows = []
        self.last_stats = {
            "fasttext_enabled": True,
            "fasttext_seed_count": len(list(seeds)),
            "fasttext_expansion_count": len(rows),
            "fasttext_budget_filled": False,
        }
        return [
            {
                **r,
                "side": side,
                "level": level,
                "source": "fasttext_cc_fr",
                "semantic_score": 0.88,
                "fasttext_score": 0.93,
                "channel": f"fasttext_{side}_L{level}",
                "content": "FastText fixture",
                "parent": "",
            }
            for r in rows
        ]


class FakePhonetic:
    """Deterministic IPA lookup and vectorized IPA encoding."""

    LOOKUP = {
        # A0 / B0 original terms
        "alpha": "IPA_AB",
        "beta": "IPA_AB",
        "charlie": "IPA_B1A0",
        "bravo": "IPA_A1B0",
        # A1 / B1 semantic candidates
        "semalpha": "IPA_A1B0",
        "sembeta": "IPA_B1A0",
        # A1 / B1 FastText candidates
        "fastalpha": "IPA_FT",
        "fastbeta": "IPA_FT",
    }

    VECTOR = {
        "IPA_AB": np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        "IPA_A1B0": np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
        "IPA_B1A0": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
        "IPA_FT": np.array([0.0, 0.0, 0.0, 1.0], dtype="float32"),
        "IPA_OTHER": np.array([0.5, 0.5, 0.5, 0.5], dtype="float32"),
    }

    def lookup_records(self, text, limit=1):
        ipa = self.LOOKUP.get(str(text))
        if not ipa:
            return []
        return [{"word": text, "ipa": ipa, "rhyme": "", "suffix2": "", "suffix3": ""}]

    def encode_ipa(self, ipa_strings):
        rows = [self.VECTOR.get(str(ipa), self.VECTOR["IPA_OTHER"]) for ipa in ipa_strings]
        return np.vstack(rows).astype("float32") if rows else np.zeros((0, 4), dtype="float32")

    def search_many(self, query_ipas, top_k=None):
        return {}


def _patch_quality_gates(module) -> None:
    """Make this a bucket-routing test, not a French-quality-filter test."""

    module.MIN_PAIR_PHONETIC = 0.99
    module.MAX_BRIDGES = 12
    module.MAX_GENERATOR_AFFORDANCES = 12
    module.MAX_IPA_CANDIDATES_PER_SIDE = 24

    module.lexically_bad_candidate_surface = lambda x: False
    module.structurally_trivial_variant = lambda a, b: False
    module.trivial_inflection_related = lambda a, b: False
    module.boring_morphophonetic_echo = lambda a, b: False
    module.universal_trivial_bridge = lambda a, b: False
    module.expression_quality = lambda item: 0.80
    module.surface_recognizability_prior = lambda surface: 0.80
    module.bridge_pivotability_score = lambda b: 0.80

    # No phonetic-neighbor expansion in this bucket-regime contract test.
    # A separate test should cover preservation/classification of expansion
    # routes. This one isolates direct A0/B0/A1/B1 bucket execution.
    if hasattr(module.BridgeMiner, "_expansion_bridges"):
        module.BridgeMiner._expansion_bridges = lambda self, *args, **kwargs: []


def _make_pipeline(module):
    pipe = object.__new__(module.RetrievalPipeline)
    pipe.expression = FakeExpression()
    pipe.phonetic = FakePhonetic()
    pipe.fasttext = FakeFastText()
    pipe.bridge_miner = module.BridgeMiner(pipe.expression, pipe.phonetic, pipe.fasttext)
    return pipe


def _bridge_pair_key(b: dict[str, Any]) -> tuple[str, str]:
    left = b.get("left_text") or b.get("sound_source") or b.get("source_surface") or b.get("a_surface")
    right = b.get("right_text") or b.get("candidate") or b.get("candidate_surface") or b.get("b_surface")
    return tuple(sorted((str(left), str(right))))


def _assert_bucket_regime_diagnostics(diag: dict[str, Any]) -> None:
    assert diag.get("retrieval_bucket_order") == EXPECTED_BUCKET_IDS

    regimes = diag.get("retrieval_bucket_regimes")
    assert regimes == EXPECTED_BUCKET_REGIMES

    pool_counts = diag.get("retrieval_bucket_pool_counts")
    assert isinstance(pool_counts, dict)
    for pool_name in ["A0", "B0", "A1", "B1"]:
        assert pool_name in pool_counts
        assert pool_counts[pool_name] > 0

    pool_surfaces = diag.get("retrieval_bucket_pool_surfaces")
    assert isinstance(pool_surfaces, dict)
    assert pool_surfaces.get("A0") == ["alpha", "charlie"]
    assert pool_surfaces.get("B0") == ["beta", "bravo"]

    a1_surfaces = set(pool_surfaces.get("A1", []))
    b1_surfaces = set(pool_surfaces.get("B1", []))

    assert "semalpha" in a1_surfaces
    assert "fastalpha" in a1_surfaces
    assert "sembeta" in b1_surfaces
    assert "fastbeta" in b1_surfaces


def _assert_final_bucket_counts(diag: dict[str, Any]) -> None:
    # Raw/pre-dedupe counts may be greater than one because the fixture
    # intentionally rediscoveres alpha/beta via semantic candidates. Final and
    # exported counts are the architecture contract here.
    raw_counts = diag.get("retrieval_bucket_counts_before_dedupe") or diag.get("retrieval_bucket_counts_before_filter")
    assert isinstance(raw_counts, dict)
    for bucket_id in EXPECTED_BUCKET_IDS:
        assert raw_counts.get(bucket_id, 0) >= 1

    after_dedupe = diag.get("retrieval_bucket_counts_after_dedupe")
    if after_dedupe is not None:
        assert after_dedupe == EXPECTED_FINAL_BUCKET_COUNTS

    after_final = diag.get("retrieval_bucket_counts_after_final_sanitize")
    assert after_final == EXPECTED_FINAL_BUCKET_COUNTS

    exported = diag.get("retrieval_exported_bucket_counts")
    assert exported == EXPECTED_FINAL_BUCKET_COUNTS


def test_bucketized_mining_routes_original_semantic_and_fasttext_candidates_to_correct_regimes():
    module = load_retrieval_module()
    _patch_quality_gates(module)

    pipe = _make_pipeline(module)

    row = pd.Series({
        "first_meaning_fr": '["alpha", "charlie"]',
        "second_meaning_fr": '["beta", "bravo"]',
        # These disallowed columns are present to ensure the input still
        # resembles the real additive TSV shape. Retrieval must ignore them.
        "text_clean": "DO_NOT_USE",
        "pun_word": "DO_NOT_USE",
        "pun_word_fr": "DO_NOT_USE",
        "first_meaning": "DO_NOT_USE",
        "second_meaning": "DO_NOT_USE",
    })

    pack = pipe.retrieve_row(row)
    bridges = pack["bridge_candidates"]
    diag = pack["bridge_diagnostics"]

    _assert_bucket_regime_diagnostics(diag)
    _assert_final_bucket_counts(diag)

    by_pair = {_bridge_pair_key(b): b for b in bridges}

    assert ("alpha", "beta") in by_pair
    assert by_pair[("alpha", "beta")]["retrieval_bucket"] == "A0_B0"

    assert ("bravo", "semalpha") in by_pair
    assert by_pair[("bravo", "semalpha")]["retrieval_bucket"] == "A1_B0"

    assert ("charlie", "sembeta") in by_pair
    assert by_pair[("charlie", "sembeta")]["retrieval_bucket"] == "B1_A0"

    assert ("fastalpha", "fastbeta") in by_pair
    assert by_pair[("fastalpha", "fastbeta")]["retrieval_bucket"] == "A1_B1"


def test_duplicate_pair_keeps_first_bucket_ownership_with_unordered_dedupe():
    module = load_retrieval_module()
    _patch_quality_gates(module)

    pipe = _make_pipeline(module)
    pack = pipe.retrieve_row(pd.Series({
        "first_meaning_fr": '["alpha", "charlie"]',
        "second_meaning_fr": '["beta", "bravo"]',
    }))

    bridges = pack["bridge_candidates"]
    alpha_beta = [
        b for b in bridges
        if _bridge_pair_key(b) == ("alpha", "beta")
    ]

    assert len(alpha_beta) == 1
    assert alpha_beta[0]["retrieval_bucket"] == "A0_B0"
    assert alpha_beta[0]["retrieval_bucket_rank"] == 1

    # Duplicate event logging is useful but not mandatory. If present, it must
    # agree with first-bucket-wins ownership.
    diag = pack["bridge_diagnostics"]
    duplicate_events = diag.get("retrieval_duplicate_pair_events", [])
    for event in duplicate_events:
        if event.get("pair_key") == alpha_beta[0].get("retrieval_pair_key"):
            assert event.get("kept_bucket") == "A0_B0"
            assert event.get("discarded_bucket") in {"A1_B0", "B1_A0", "A1_B1"}


def test_retrieve_row_exposes_generator_bucket_fields():
    module = load_retrieval_module()
    _patch_quality_gates(module)

    pipe = _make_pipeline(module)
    pack = pipe.retrieve_row(pd.Series({
        "first_meaning_fr": '["alpha", "charlie"]',
        "second_meaning_fr": '["beta", "bravo"]',
    }))

    gen_pack = pack["generator_affordance_pack"]
    exported = gen_pack["top_bridge_candidates"]

    assert exported
    for item in exported:
        assert item["retrieval_bucket"] in EXPECTED_BUCKET_IDS
        assert isinstance(item["retrieval_bucket_rank"], int)
        assert item["retrieval_bucket_rank"] >= 1


def test_retrieval_columns_preserve_bucket_diagnostics_and_affordance_buckets():
    module = load_retrieval_module()
    _patch_quality_gates(module)

    pipe = _make_pipeline(module)
    pack = pipe.retrieve_row(pd.Series({
        "first_meaning_fr": '["alpha", "charlie"]',
        "second_meaning_fr": '["beta", "bravo"]',
    }))

    assert hasattr(module, "_retrieval_columns_from_pack"), (
        "Bucketization must survive into the normal TSV row-output path via "
        "_retrieval_columns_from_pack(pack)."
    )

    cols = module._retrieval_columns_from_pack(pack)

    assert "retrieval_bucket_counts_json" in cols
    assert "retrieval_affordance_buckets_json" in cols

    bucket_counts = json.loads(cols["retrieval_bucket_counts_json"])
    assert bucket_counts == EXPECTED_FINAL_BUCKET_COUNTS

    affordance_buckets = json.loads(cols["retrieval_affordance_buckets_json"])
    assert list(affordance_buckets.keys()) == EXPECTED_BUCKET_IDS

    for bucket_id in EXPECTED_BUCKET_IDS:
        assert affordance_buckets[bucket_id], bucket_id
        for item in affordance_buckets[bucket_id]:
            assert item["retrieval_bucket"] == bucket_id
            assert isinstance(item["retrieval_bucket_rank"], int)
