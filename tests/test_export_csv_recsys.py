"""export_csv.py: Fix 9's similarity-recommender columns.

Follows tests/test_persona_attributes.py's own pattern for a new optional
bank column: present but empty for a run that never wrote it, present and
populated for one that did.
"""

from __future__ import annotations

import json

import export_csv


def _write_interaction(run_dir, **overrides) -> dict:
    base = dict(
        interaction_id="i1", run_id="r", turn=1, speaker_agent_id="agent_0000",
        neighbor_agent_ids=["agent_0001"], stance_before=4.0, stance_after=5.0,
        reason_text="because", content_type="bank_item", prompt_token_count=10,
        completion_token_count=5, model_backend_id="gpt-4o", latency_ms=1,
        api_call_status="success", timestamp_utc="2026-01-01T00:00:00Z",
        items_read=["item_1"],
    )
    base.update(overrides)
    return base


def _run_dir(tmp_path, name, rows, config=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps(config or {"run_id": name}))
    (run_dir / "interactions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return run_dir


def test_recsys_columns_are_populated_when_present(tmp_path):
    run_dir = _run_dir(
        tmp_path, "run1",
        [_write_interaction(
            tmp_path / "run1", feed_size=2, n_neighbors_over_threshold=1,
            recsys_fallback=False,
        )],
        config={"run_id": "run1", "recsys_threshold": 0.5},
    )
    df = export_csv.derive(export_csv.mark_duplicates(export_csv.load_run(run_dir)))
    assert df.loc[0, "feed_size"] == 2
    assert df.loc[0, "n_neighbors_over_threshold"] == 1
    assert bool(df.loc[0, "recsys_fallback"]) is False
    assert df.loc[0, "recsys_threshold"] == 0.5


def test_recsys_columns_are_empty_rather_than_absent_when_never_written(tmp_path):
    """Every run before Fix 9, and every non-graph_similarity strategy."""
    run_dir = _run_dir(tmp_path, "run2", [_write_interaction(tmp_path / "run2")])
    df = export_csv.derive(export_csv.mark_duplicates(export_csv.load_run(run_dir)))
    assert "feed_size" in df.columns
    assert "n_neighbors_over_threshold" in df.columns
    assert "recsys_fallback" in df.columns
    assert "recsys_threshold" in df.columns
    import pandas as pd
    assert pd.isna(df.loc[0, "feed_size"])
    assert pd.isna(df.loc[0, "n_neighbors_over_threshold"])
    assert df.loc[0, "recsys_fallback"] is None
    assert df.loc[0, "recsys_threshold"] is None


def test_no_feed_status_counts_as_valid_for_analysis(tmp_path):
    run_dir = _run_dir(
        tmp_path, "run3",
        [_write_interaction(
            tmp_path / "run3", api_call_status="no_feed", neighbor_agent_ids=[],
            items_read=[], reason_text="", feed_size=0, n_neighbors_over_threshold=2,
            recsys_fallback=False,
        )],
    )
    df = export_csv.derive(export_csv.mark_duplicates(export_csv.load_run(run_dir)))
    assert bool(df.loc[0, "is_valid_for_analysis"]) is True
