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
