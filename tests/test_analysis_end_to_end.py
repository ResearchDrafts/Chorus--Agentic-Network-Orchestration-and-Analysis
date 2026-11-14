"""The analysis tier against runs the orchestrator actually produced.

The unit tests in test_analysis_metrics.py check the rules on hand-built
inputs. This checks the seam: that a real run directory -- interactions.jsonl,
graph_edges.csv, run_config.json, written by the orchestrator through the
export's own loading pipeline -- reaches those rules with the columns they
expect. Every bug this catches is a column name or a dtype, which no amount
of arithmetic testing would find.

Only the LLM calls are faked.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis import metrics as m
from analysis import stats as st
from sandbox.config_loader import load_run_config
from sandbox.simulation_orchestrator import build_orchestrator
from tests.factories import make_logprob_response

CAMPAIGN = Path("configs/rq1/campaign")
ARMS = ("english", "hindi")
SEEDS = (1, 2)


@pytest.fixture
def campaign_runs(tmp_path, bank_gateway):
    """Two arms x two experiments x two seeds, actually run.

    The probe reading is a seeded draw over the scale rather than a constant,
    so agents genuinely move and every outcome class has a chance to appear --
    a constant reading would make every round's dx zero and leave the whole
    classification untested at this level.
    """

    async def _run(**overrides) -> Path:
        rng = random.Random(11)
        bank_gateway.score_next_token.side_effect = lambda _: make_logprob_response(
            probabilities={rng.choice([1, 2, 3, 4, 5, 6, 7]): 1.0}
        )
        runs_dir = tmp_path / "runs"
        for seed_index in SEEDS:
            for experiment in ("e3", "e4"):
                for language in ARMS:
                    stem = f"{experiment}_{language}_ws_s{seed_index:02d}"
                    raw = yaml.safe_load((CAMPAIGN / f"{stem}.yaml").read_text())
                    raw.update(overrides)
                    path = tmp_path / f"{stem}.yaml"
                    path.write_text(
                        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
                    )
                    run = load_run_config(path)
                    await build_orchestrator(
                        run, bank_gateway, runs_dir=runs_dir
                    ).run()
        return runs_dir

    return _run


@pytest.fixture
def tau_file(tmp_path):
    path = tmp_path / "tau.json"
    path.write_text(json.dumps({
        "percentile": 95, "smoke": False,
        "languages": {
            language: {"tau": 0.1, "tau_conf": 0.1, "n_ok": 400}
            for language in ("english", "hinglish", "romanized_hindi", "hindi")
        },
    }))
    return path


EPSILON = ["--epsilon", "var_x=0.02", "extremity=0.02", "assortativity=0.05"]


async def test_metrics_reads_a_real_campaign(tmp_path, campaign_runs, tau_file):
    runs_dir = await campaign_runs()
    out = tmp_path / "results"

    assert m.main([
        "--runs-dir", str(runs_dir), "--tau", str(tau_file),
        "--out", str(out), *EPSILON,
    ]) == 0

    runs = pd.read_csv(out / "run_metrics.csv")
    assert len(runs) == len(ARMS) * 2 * len(SEEDS)
    assert set(runs.experiment) == {"e3", "e4"}
    assert set(runs.language) == set(ARMS)
    # Seed index i got base_seed + i, shared by every arm of that set.
    assert sorted(runs.seed.unique()) == [1001, 1002]
    assert list(runs.columns) == list(m.RUN_COLUMNS)

    rounds = pd.read_csv(out / "round_metrics.csv")
    # K rounds plus round 0, per run.
    assert len(rounds) == len(runs) * (int(runs.K.iloc[0]) + 1)
    assert list(rounds.columns) == list(m.ROUND_COLUMNS)

    probes = pd.read_csv(out / "probes.csv")
    assert len(probes) == len(rounds) * int(runs.M.iloc[0])
    assert list(probes.columns) == list(m.PROBE_COLUMNS)


async def test_the_outcome_classes_actually_fire_on_a_real_run(
    tmp_path, campaign_runs, tau_file
):
    """A test that only checked the columns would pass against an analysis
    that classified everything as `none`."""
    runs_dir = await campaign_runs()
    out = tmp_path / "results"
    m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
            "--out", str(out), *EPSILON])

    probes = pd.read_csv(out / "probes.csv")
    classified = probes[probes.turn > 0]
    assert classified.outcome.notna().all()
    assert set(classified.outcome) & {"flip", "harden", "soften"}
    # Round 0 is the reading before anything was read, so it has no outcome.
    assert probes[probes.turn == 0].outcome.isna().all()
    assert probes[probes.turn == 0].dx.isna().all()


async def test_assortativity_comes_from_the_runs_own_graph_file(
    tmp_path, campaign_runs, tau_file
):
    """Read from graph_edges.csv rather than rebuilt from the config, so the
    metric does not depend on this networkx version producing the same graph
    from the same seed across a reinstall."""
    runs_dir = await campaign_runs()
    assert (runs_dir / "e3_english_ws_s01" / "graph_edges.csv").exists()
    out = tmp_path / "results"
    m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
            "--out", str(out), *EPSILON])

    rounds = pd.read_csv(out / "round_metrics.csv")
    assert rounds.assortativity.notna().any()

    # Remove the graph and the column becomes null rather than zero: "agents
    # are not adjacent to their own side" and "there is no adjacency" are
    # different facts.
    for run_dir in runs_dir.iterdir():
        (run_dir / "graph_edges.csv").unlink()
    m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
            "--out", str(out), *EPSILON])
    assert pd.read_csv(out / "round_metrics.csv").assortativity.isna().all()


async def test_epsilon_from_e4_uses_only_the_control_runs(
    tmp_path, campaign_runs, tau_file, capsys
):
    runs_dir = await campaign_runs()
    out = tmp_path / "results"
    assert m.main([
        "--runs-dir", str(runs_dir), "--tau", str(tau_file),
        "--out", str(out), "--epsilon-from-e4",
    ]) == 0
    assert "ε from the E4 control arm" in capsys.readouterr().out


async def test_an_uncalibrated_arm_stops_the_analysis(
    tmp_path, campaign_runs
):
    """Rather than classifying its movements against another arm's noise,
    which would look exactly like a calibrated result."""
    runs_dir = await campaign_runs()
    partial = tmp_path / "partial_tau.json"
    partial.write_text(json.dumps({
        "languages": {"english": {"tau": 0.1, "tau_conf": 0.1, "n_ok": 400}}
    }))
    with pytest.raises(m.MissingTau, match="hindi"):
        m.main(["--runs-dir", str(runs_dir), "--tau", str(partial),
                "--out", str(tmp_path / "results"), *EPSILON])


async def test_a_generated_regime_run_is_skipped(
    tmp_path, campaign_runs, tau_file, mock_gateway, capsys
):
    """A generated run has no probe, so it has no x, and every metric here is
    defined on x. Skipped with a note rather than silently."""
    runs_dir = await campaign_runs()

    raw = yaml.safe_load(Path("configs/example_run.yaml").read_text())
    raw.update(M=5, N=2, K=2, run_id="a_generated_run",
               persona_pool_id="test_pool")
    raw.pop("max_cost_usd", None)
    path = tmp_path / "generated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run = load_run_config(path)
    await build_orchestrator(run, mock_gateway, runs_dir=runs_dir).run()

    out = tmp_path / "results"
    assert m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
                   "--out", str(out), *EPSILON]) == 0
    assert "skipped a_generated_run: not a bank-regime run" in capsys.readouterr().out
    assert "a_generated_run" not in set(pd.read_csv(out / "run_metrics.csv").run_id)


async def test_a_failed_round_is_counted_not_classified(
    tmp_path, campaign_runs, tau_file, bank_gateway
):
    """A round whose probe fell below the mass floor was logged and never
    applied, so it carries no reading -- and n_failed says so rather than the
    row quietly disappearing."""
    runs_dir = await campaign_runs()
    # Re-run one arm with every probe off-scale, into its own directory.
    bank_gateway.score_next_token.side_effect = None
    bank_gateway.score_next_token.return_value = make_logprob_response(
        probabilities={5: 1.0}, off_scale=0.9
    )
    raw = yaml.safe_load((CAMPAIGN / "e3_english_ws_s03.yaml").read_text())
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    await build_orchestrator(
        load_run_config(path), bank_gateway, runs_dir=runs_dir
    ).run()

    out = tmp_path / "results"
    assert m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
                   "--out", str(out), *EPSILON]) == 0
    # The run produced no valid reading at all, so it contributes no rows
    # rather than rows of nulls.
    runs = pd.read_csv(out / "run_metrics.csv")
    assert "e3_english_ws_s03" not in set(runs.run_id)


async def test_the_whole_pipeline_runs_end_to_end(
    tmp_path, campaign_runs, tau_file
):
    """metrics -> stats -> figures, on output a real run produced."""
    runs_dir = await campaign_runs()
    out = tmp_path / "results"
    assert m.main(["--runs-dir", str(runs_dir), "--tau", str(tau_file),
                   "--out", str(out), "--epsilon-from-e4"]) == 0

    # stats needs an E1 sweep; the network half can run without one.
    assert st.main([
        "--e1", str(tmp_path / "absent.jsonl"),
        "--results", str(out), "--out", str(out), "--skip-e1",
    ]) == 0
    network = pd.read_csv(out / "network_contrasts.csv")
    assert set(network.language) == {"hindi"}
    assert set(network.scope) == {"within_e3", "within_e4", "did"}
    assert (network.n == len(SEEDS)).all()
    # Two seeds is far below what a signed-rank test can resolve, and the
    # column says so rather than leaving the reader to work it out.
    assert network.underpowered.all()
