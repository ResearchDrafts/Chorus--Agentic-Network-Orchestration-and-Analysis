"""Post-hoc analysis of a completed RQ1 campaign.

Offline, and strictly downstream: every module here reads a finished run's
`interactions.jsonl`, `graph_edges.csv` and `run_config.json`, and nothing
here is imported by the orchestrator or by anything a run touches. So a
change to an analysis cannot change a dataset, and re-deriving a result is
always free.

Three stages, in order:

    python -m analysis.metrics  --runs-dir <dir> --tau calibration/tau.json \\
                                --out results/ --epsilon-from-e4
    python -m analysis.stats    --e1 e1/trials.jsonl --results results/ \\
                                --out results/
    python -m analysis.figures  --results results/ --e1 e1/trials.jsonl \\
                                --calibration calibration/ --out results/figures/

`metrics` turns runs into three tidy tables, classifying every round-to-round
change against the τ that `scripts/calibrate_tau.py` measured. `stats` fits
the models and reports the contrasts. `figures` draws them. Each reads only
what the stage before it wrote, so any one can be re-run on its own.

The reading `x` is `probe_scorer.to_x(probe_mean)` -- one definition, shared
with `export_csv.py` and the single-exposure scripts.
"""
