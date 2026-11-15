"""Checkpoint save/load (full_design_doc.md Sec 3.14).

Per-turn checkpoint of full agent state, so a failed run can resume rather
than restart. Overwrites the previous checkpoint each turn (not
append-only, unlike interactions.jsonl -- only the latest checkpoint is
ever needed for resume). Uses a write-to-.tmp-then-atomic-rename pattern
(Path.replace, atomic on POSIX) so a mid-write crash never leaves a torn
checkpoint file behind.
"""

from __future__ import annotations

from pathlib import Path

from sandbox.models import Agent, CheckpointState


class CheckpointManager:
    def __init__(self, runs_dir: Path = Path("runs")):
        self._runs_dir = runs_dir

    def save(
        self,
        run_id: str,
        turn: int,
        agent_snapshot: list[Agent],
        graph_edges: list[tuple[str, str]] | None = None,
    ) -> None:
        checkpoint_dir = self._runs_dir / run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = CheckpointState(
            run_id=run_id,
            last_completed_turn=turn,
            agent_snapshot=agent_snapshot,
            graph_edges=graph_edges,
        )
        tmp_path = checkpoint_dir / "latest.json.tmp"
        final_path = checkpoint_dir / "latest.json"
        tmp_path.write_text(state.model_dump_json())
        tmp_path.replace(final_path)

    def load(self, run_id: str) -> CheckpointState | None:
        """Returns None if no checkpoint exists for this run_id (fresh
        run). A checkpoint that exists but fails schema validation raises
        pydantic.ValidationError uncaught -- treated as a fatal resume
        failure requiring manual intervention, never silently falling back
        to turn 0, which would discard completed work and re-spend budget."""
        path = self._runs_dir / run_id / "checkpoints" / "latest.json"
        if not path.exists():
            return None
        return CheckpointState.model_validate_json(path.read_text())
