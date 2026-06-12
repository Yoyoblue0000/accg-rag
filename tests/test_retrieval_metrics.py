# -*- coding: utf-8 -*-
"""候选检索基线指标测试。"""

from mini_agent.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_candidates,
    extract_provisional_gold,
)


def test_extract_provisional_gold_from_reference_answer():
    answer = (
        "The function `get_environ_proxies` in `src/requests/utils.py` "
        "delegates to `Session.resolve_redirects()`."
    )

    gold = extract_provisional_gold(answer)

    assert gold.paths == ["src/requests/utils.py"]
    assert "get_environ_proxies" in gold.symbols
    assert "session.resolve_redirects" in gold.symbols


def test_retrieval_metrics_use_binary_candidate_relevance():
    gold = extract_provisional_gold(
        "`get_environ_proxies` is defined in `src/requests/utils.py`."
    )
    candidates = [
        {
            "id": "src/requests/sessions.py::merge_setting",
            "name": "merge_setting",
            "file": "src/requests/sessions.py",
        },
        {
            "id": "src/requests/utils.py::get_environ_proxies",
            "name": "get_environ_proxies",
            "file": "src/requests/utils.py",
        },
    ]

    metrics = evaluate_candidates(candidates, gold)

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["binary_relevance"] is True


def test_aggregate_metrics_reports_latency_separately():
    summary = aggregate_retrieval_metrics([
        {
            "retrieval": {"status": "ok", "duration_ms": 10.0},
            "retrieval_metrics": {
                "evaluable": True,
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "recall_at_10": 1.0,
                "mrr": 1.0,
                "ndcg_at_1": 1.0,
                "ndcg_at_3": 1.0,
                "ndcg_at_5": 1.0,
                "ndcg_at_10": 1.0,
            },
        },
        {
            "retrieval": {"status": "fallback", "duration_ms": 30.0},
            "retrieval_metrics": {
                "evaluable": True,
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "ndcg_at_1": 0.0,
                "ndcg_at_3": 0.0,
                "ndcg_at_5": 0.0,
                "ndcg_at_10": 0.0,
            },
        },
    ])

    assert summary["latency_ms_mean"] == 20.0
    assert summary["latency_ms_p50"] == 20.0
    assert summary["latency_ms_p95"] == 30.0
    assert summary["fallbacks"] == 1
