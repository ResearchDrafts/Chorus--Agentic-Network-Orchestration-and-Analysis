"""analysis/figures.py.

This file checks that the figures are written, and that the two things that
would be wrong without a reader noticing are right: the colour assignment is
fixed per language rather than cycled, and an arm with no colour raises
instead of silently sharing one. What the figures LOOK like is not something
a test can judge -- that is what rendering them and looking is for.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from analysis import figures as fg

LANGUAGES = ("english", "hinglish", "romanized_hindi", "hindi")


@pytest.fixture
def inputs(tmp_path):
    """A tiny but complete set of the three files the figures read."""
    results = tmp_path / "results"
    results.mkdir()

    rounds, probes = [], []
    for index, language in enumerate(LANGUAGES):
        for experiment in ("e3", "e4"):
            for seed in (1001, 1002):
                for turn in range(4):
                    rounds.append({
                        "run_id": f"{experiment}_{language}_ws_s{seed - 1000:02d}",
                        "experiment": experiment, "language": language, "seed": seed,
                        "turn": turn,
                        "mean_x": 0.05 * index + 0.02 * turn,
                        "var_x": 0.10 + 0.01 * turn,
                        "extremity": 0.1, "assortativity": 0.2,
                        "n_valid": 4, "n_failed": 0,
                        "mean_reply_prompt_tokens": None if turn == 0 else 600 + 50 * index + 20 * turn,
                        "mean_probe_prompt_tokens": 400 + 30 * index,
                    })
                    for agent in range(4):
                        probes.append({
                            "run_id": f"{experiment}_{language}_ws_s{seed - 1000:02d}",
                            "experiment": experiment, "language": language,
                            "seed": seed, "turn": turn, "agent_id": f"agent_{agent:04d}",
                            "x": 0.1 * agent - 0.15, "side_conf": 0.7,
                            "probe_mass": 0.9 - 0.02 * index,
                            "reply_prompt_tokens": 0 if turn == 0 else 600 + 50 * index,
                            "probe_prompt_tokens": 400 + 30 * index,
                        })
    pd.DataFrame(rounds).to_csv(results / "round_metrics.csv", index=False)
    pd.DataFrame(probes).to_csv(results / "probes.csv", index=False)

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "tau.json").write_text(json.dumps({
        "percentile": 95, "smoke": False,
        "languages": {
            language: {"tau": 0.08 + 0.01 * i, "tau_conf": 0.1, "n_ok": 400}
            for i, language in enumerate(LANGUAGES)
        },
    }))

    e1 = tmp_path / "trials.jsonl"
    rows = []
    for item in range(4):
        for language in LANGUAGES:
            for persona in range(3):
                rows.append({
                    "key": f"e1|{language}|bank_t{item:02d}|p{persona:03d}|lean4",
                    "language": language, "item_id": f"bank_t{item:02d}",
                    "arg_id": f"arg_{item}",
                    "strength": "strong" if item < 2 else "weak",
                    "side": "pro", "persona_index": persona, "lean": 4,
                    "toward": 0.2 - 0.03 * item, "status": "ok",
                })
    e1.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    return results, e1, calibration, tmp_path / "figures"


def test_all_four_figures_are_written_as_pdf_and_png(inputs):
    results, e1, calibration, out = inputs
    assert fg.main([
        "--results", str(results), "--e1", str(e1),
        "--calibration", str(calibration), "--out", str(out),
    ]) == 0

    for stem in ("trajectories", "e1_toward", "tau", "instrument"):
        for suffix in ("pdf", "png"):
            path = out / f"{stem}.{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 1000, path


def test_one_arm_alone_still_renders(inputs):
    """A campaign stopped after the English arm must still produce figures:
    the point of looking at them early is to catch a problem before spending
    on the other three."""
    results, e1, calibration, out = inputs
    for name in ("round_metrics", "probes"):
        frame = pd.read_csv(results / f"{name}.csv")
        frame[frame.language == "english"].to_csv(results / f"{name}.csv", index=False)

    assert fg.main([
        "--results", str(results), "--e1", str(e1),
        "--calibration", str(calibration), "--out", str(out),
    ]) == 0
    assert (out / "trajectories.png").exists()


def test_the_colour_follows_the_language_not_its_position():
    """A filter that drops an arm must not repaint the survivors."""
    assert fg.color_for("hindi") == fg.LANGUAGE_COLORS["hindi"]
    assert len(set(fg.LANGUAGE_COLORS.values())) == len(fg.LANGUAGE_COLORS)
    assert set(fg.LANGUAGE_COLORS) == set(LANGUAGES)


def test_an_unknown_language_raises_rather_than_sharing_a_hue():
    with pytest.raises(KeyError, match="no colour assigned"):
        fg.color_for("tamil")


def test_english_is_listed_first():
    """Every other arm is reported as a difference from it."""
    frame = pd.DataFrame({"language": ["hindi", "english", "hinglish"]})
    assert fg.languages_in(frame) == ["english", "hinglish", "hindi"]


# --- Fix 6: e7's style, and the new per-round line figures -----------------


def test_e7_has_its_own_style_and_label():
    """e7 (rewiring without selection) must not share e6's dash style, or
    it would read as a delivery-off control rather than a no-selection one."""
    assert set(fg.ECHO_EXPERIMENT_STYLES) == {"e5", "e6", "e7"}
    assert len(set(fg.ECHO_EXPERIMENT_STYLES.values())) == 3
    assert fg.ECHO_EXPERIMENT_STYLES["e7"] == ":"
    assert "e7" in fg.ECHO_EXPERIMENT_LABELS


def _echo_rounds() -> pd.DataFrame:
    rows = []
    for language in LANGUAGES:
        for experiment in ("e5", "e6", "e7"):
            for seed in (1001, 1002):
                for turn in range(3):
                    rows.append({
                        "run_id": f"{experiment}_{language}_ws_s{seed - 1000:02d}",
                        "experiment": experiment, "language": language,
                        "seed": seed, "turn": turn,
                        "assort_excess": 0.1 * turn,
                        "cross_side_edge_share": 0.5 - 0.1 * turn,
                        "nci_excess": 0.05 * turn,
                        "disagreement_ratio": 1.0 - 0.1 * turn,
                        "minority_share": 0.3,
                        "n_side_switches": None if turn == 0 else 2,
                    })
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "func, column",
    [
        (fg.figure_nci_excess, "nci_excess"),
        (fg.figure_disagreement_ratio, "disagreement_ratio"),
        (fg.figure_minority_share, "minority_share"),
        (fg.figure_n_side_switches, "n_side_switches"),
    ],
)
def test_new_line_figures_render_with_e5_e6_e7(tmp_path, func, column):
    rounds = _echo_rounds()
    paths = func(rounds, tmp_path)
    assert paths
    for path in paths:
        assert path.exists() and path.stat().st_size > 500


def test_figures_main_writes_the_new_line_figures_when_columns_present(inputs):
    """round_metrics.csv from a campaign with E5/E6/E7 columns must produce
    all four new figures; main() guards each on column presence so an
    e3/e4-only campaign (the `inputs` fixture, unmodified) draws none of
    them without erroring."""
    results, e1, calibration, out = inputs
    rounds = pd.read_csv(results / "round_metrics.csv")
    rounds["nci_excess"] = 0.1
    rounds["disagreement_ratio"] = 0.9
    rounds["minority_share"] = 0.3
    rounds["n_side_switches"] = 1
    rounds.to_csv(results / "round_metrics.csv", index=False)

    assert fg.main([
        "--results", str(results), "--e1", str(e1),
        "--calibration", str(calibration), "--out", str(out),
    ]) == 0
    for stem in ("nci_excess", "disagreement_ratio", "minority_share", "n_side_switches"):
        assert (out / f"{stem}.png").exists()


# --- the network-snapshot layout and per-experiment files -------------------


def test_spring_layout_uses_the_wider_k_and_more_iterations(monkeypatch):
    """k = 1/sqrt(n) and iterations=300, not networkx's own defaults --
    separated camps need the extra room to actually pull apart rather than
    settling into one blob."""
    import networkx as nx

    captured = {}
    real_spring_layout = nx.spring_layout

    def spy(graph, **kwargs):
        captured.update(kwargs)
        return real_spring_layout(graph, **kwargs)

    monkeypatch.setattr(fg.nx, "spring_layout", spy)
    graph_edges = [("a", "b"), ("b", "c"), ("c", "a")]
    fig, axis = fg.plt.subplots()
    fg._draw_one_snapshot(axis, graph_edges, {"a": 1.0, "b": 1.0, "c": -1.0})
    fg.plt.close(fig)

    assert captured["iterations"] == 300
    assert captured["k"] == pytest.approx(1.0 / (3 ** 0.5))
    assert captured["seed"] == fg.FIGURE_LAYOUT_SEED


def _write_run_dir(run_dir, edges, x_by_agent, k=2):
    run_dir.mkdir(parents=True)
    (run_dir / "graph_edges.csv").write_text(
        "agent_a,agent_b\n" + "\n".join(f"{a},{b}" for a, b in edges) + "\n",
        encoding="utf-8",
    )


def test_figure_network_snapshots_writes_its_own_stem(tmp_path):
    graphs = fg.RunGraphs.__new__(fg.RunGraphs)
    graphs.run_id = "e6_english_ws_s01"
    graphs.k = 2
    graphs._static = __import__("networkx").Graph([("agent_0000", "agent_0001")])
    graphs._by_round = None
    run_graphs = {"english": graphs}
    probes = pd.DataFrame([
        {"run_id": "e6_english_ws_s01", "turn": turn, "agent_id": agent, "x": 0.5}
        for turn in (0, 1, 2)
        for agent in ("agent_0000", "agent_0001")
    ])
    paths = fg.figure_network_snapshots(
        probes, run_graphs, tmp_path, stem="network_snapshots_e6", experiment_label="E6 (control)"
    )
    assert paths
    assert all(p.stem == "network_snapshots_e6" for p in paths)


def test_main_writes_one_snapshot_file_per_echo_experiment_present(tmp_path):
    """e5, e6 and e7 each get their own network_snapshots file -- never one
    overwriting the other -- when all three have runs on disk."""
    results = tmp_path / "results"
    results.mkdir()
    runs_dir = tmp_path / "runs"

    rounds, probes, run_rows = [], [], []
    for experiment in ("e5", "e6", "e7"):
        run_id = f"{experiment}_english_ws_s01"
        run_rows.append({
            "run_id": run_id, "experiment": experiment, "language": "english",
            "seed": 1001, "K": 2,
        })
        _write_run_dir(
            runs_dir / run_id,
            edges=[("agent_0000", "agent_0001")],
            x_by_agent={"agent_0000": 1.0, "agent_0001": -1.0},
        )
        for turn in range(3):
            rounds.append({
                "run_id": run_id, "experiment": experiment, "language": "english",
                "seed": 1001, "turn": turn, "mean_x": 0.0, "var_x": 0.1,
                "extremity": 0.1, "assortativity": 0.1, "n_valid": 2, "n_failed": 0,
                "mean_reply_prompt_tokens": 500, "mean_probe_prompt_tokens": 300,
            })
            for agent, x in (("agent_0000", 1.0), ("agent_0001", -1.0)):
                probes.append({
                    "run_id": run_id, "experiment": experiment, "language": "english",
                    "seed": 1001, "turn": turn, "agent_id": agent, "x": x,
                    "side_conf": 0.8, "probe_mass": 0.9,
                    "reply_prompt_tokens": 500, "probe_prompt_tokens": 300,
                })

    pd.DataFrame(rounds).to_csv(results / "round_metrics.csv", index=False)
    pd.DataFrame(probes).to_csv(results / "probes.csv", index=False)
    pd.DataFrame(run_rows).to_csv(results / "run_metrics.csv", index=False)

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "tau.json").write_text(json.dumps({
        "percentile": 95, "smoke": False,
        "languages": {"english": {"tau": 0.08, "tau_conf": 0.1, "n_ok": 10}},
    }))
    e1 = tmp_path / "trials.jsonl"
    e1.write_text(json.dumps({
        "key": "e1|english|bank_t00|p000|lean4", "language": "english",
        "item_id": "bank_t00", "arg_id": "arg_0", "strength": "strong",
        "side": "pro", "persona_index": 0, "lean": 4, "toward": 0.1, "status": "ok",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "figures"

    assert fg.main([
        "--results", str(results), "--e1", str(e1), "--calibration", str(calibration),
        "--out", str(out), "--runs-dir", str(runs_dir),
    ]) == 0

    assert (out / "network_snapshots.png").exists()        # e5, unsuffixed
    assert (out / "network_snapshots_e6.png").exists()
    assert (out / "network_snapshots_e7.png").exists()
