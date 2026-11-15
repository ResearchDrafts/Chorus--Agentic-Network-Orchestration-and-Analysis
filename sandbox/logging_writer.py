"""Append-only persistence (full_design_doc.md Sec 3.8).

One run = one directory, containing run_config.json, interactions.jsonl,
agents_final.jsonl and -- when the run asks for it -- prompts.jsonl. Disk-write failures (OSError, e.g. disk full)
propagate uncaught -- treated as fatal, since silently losing an
Interaction record would corrupt the run's data integrity in a way no
downstream analysis could detect.

run_dir creation is NOT this module's responsibility: per the design doc's
own sample code, LoggingWriter never calls run_dir.mkdir() (unlike
checkpoint_manager.py, which does create its own checkpoints/ directory).
The Orchestrator is expected to create run_dir before constructing a
LoggingWriter; calling any write method against a run_dir that doesn't
exist yet propagates FileNotFoundError uncaught, exactly like any other
disk-write failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sandbox.models import Agent, ExperimentRun, Interaction, PromptRecord, RewireRecord


class LoggingWriter:
    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._interactions_path = run_dir / "interactions.jsonl"
        self._prompts_path = run_dir / "prompts.jsonl"
        self._rewires_path = run_dir / "rewires.jsonl"
        self._graph_edges_by_round_path = run_dir / "graph_edges_by_round.csv"
        self._lock = asyncio.Lock()  # serializes concurrent appends from dispatch tasks

    async def write_interaction(self, interaction: Interaction) -> None:
        """
        Appends one JSON line. Uses an asyncio.Lock rather than relying on
        OS-level atomic append, because Python's async file writes are not
        guaranteed atomic across concurrent tasks writing to the same file
        handle within one process (this is a single-process design -- no
        cross-process write contention to handle).
        """
        line = interaction.model_dump_json() + "\n"
        async with self._lock:
            with open(self._interactions_path, "a", encoding="utf-8") as f:
                f.write(line)

    async def write_prompt(self, record: PromptRecord) -> None:
        """Appends one line to prompts.jsonl: one LLM call, verbatim.

        Shares `_lock` with write_interaction rather than taking one of its
        own. The two files are written from the same concurrent dispatch
        tasks, the lock is held only across a single small append, and one
        lock is one fewer ordering to reason about -- while two independent
        locks would let a task holding one wait on the other for no benefit.

        **Duplicate lines are possible, and acceptable.** A resumed run
        re-runs the round its checkpoint does not cover, so that round's
        prompts are appended a second time. Key on (turn, agent_id, call)
        with the last write winning, exactly as export_csv.py does for
        interactions.jsonl; the later copy is the call whose result the
        resumed run actually used. Deduplicating here would mean reading the
        whole file back on every append.
        """
        line = record.model_dump_json() + "\n"
        async with self._lock:
            with open(self._prompts_path, "a", encoding="utf-8") as f:
                f.write(line)

    async def write_rewire(self, record: RewireRecord) -> None:
        """Appends one line to rewires.jsonl: one read post's rewire
        decision, applied or skipped.

        Called only after a round has fully committed (see
        SimulationOrchestrator._rewire_round), never from inside Phase A or
        Phase B's gather, so -- unlike interactions.jsonl and prompts.jsonl --
        a crash mid-round leaves nothing here for that round at all: the
        resumed pass re-runs the round and writes it exactly once, with no
        duplicate-row problem to resolve on read.
        """
        line = record.model_dump_json() + "\n"
        async with self._lock:
            with open(self._rewires_path, "a", encoding="utf-8") as f:
                f.write(line)

    def write_graph_edges_round(self, turn: int, edges: list[tuple[str, str]]) -> None:
        """Appends one round's block to graph_edges_by_round.csv: the edges
        this run used to choose that round's neighbours.

        Synchronous and unlocked, like write_run_config and
        write_agents_final: called once per round from the orchestrator's
        own sequential rewiring pass, never from a concurrently-dispatched
        task, so there is no write it could race with.

        Writes the header only when the file does not yet exist, so a
        resumed run's append picks up right after the rows an earlier pass
        already wrote.
        """
        is_new = not self._graph_edges_by_round_path.exists()
        with open(self._graph_edges_by_round_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("turn,agent_a,agent_b\n")
            for a, b in edges:
                f.write(f"{turn},{a},{b}\n")

    def write_run_config(self, run: ExperimentRun) -> None:
        """Called once at run start, before any concurrent activity -- no
        locking needed."""
        path = self._run_dir / "run_config.json"
        path.write_text(run.model_dump_json(indent=2))

    def write_agents_final(self, agents: list[Agent]) -> None:
        """Called once at run completion, after all turns finish and no
        further concurrent writes are possible -- no locking needed.
        Overwrites (open mode "w"), not appends: a second call with a
        different agent list leaves only that second list's contents."""
        path = self._run_dir / "agents_final.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for agent in agents:
                f.write(agent.model_dump_json() + "\n")
