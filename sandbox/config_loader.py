"""Feature 1: Configuration Loading & Validation.

Parses and validates a run config into an ExperimentRun before any other
feature is permitted to act. Fails loudly and immediately, before a single
dollar is spent, per sandbox_feature_specs.md's Feature 1 Problem Statement.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox.models import ExperimentRun


class ConfigLoadError(Exception):
    """Raised for any config problem; wraps the underlying cause."""

    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        super().__init__(f"Invalid config at {path}: {'; '.join(errors)}")


def load_run_config(path: Path) -> ExperimentRun:
    """
    Reads a YAML file, validates it against the ExperimentRun schema, and
    resolves referenced assets (meme_pool_id, persona_pool_id) to confirm
    they exist on disk -- without loading their full contents.

    Raises ConfigLoadError on any failure. Never returns a partially valid
    ExperimentRun.
    """
    if not path.exists():
        raise ConfigLoadError(path, [f"file not found: {path}"])

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigLoadError(path, [f"malformed YAML: {e}"]) from e

    if raw is None:
        raise ConfigLoadError(path, ["config file is empty"])

    # A syntactically valid YAML document need not be a mapping: "- a\n- b",
    # "just_a_string", and "42" all parse cleanly into a list/str/int. Without
    # this guard the raw.pop() below raises TypeError/AttributeError, breaking
    # this function's documented contract that every failure surfaces as a
    # ConfigLoadError.
    if not isinstance(raw, dict):
        raise ConfigLoadError(
            path, [f"config root must be a mapping, got {type(raw).__name__}"]
        )

    # git_commit_hash is populated here, never user-supplied in the YAML
    # (Feature 1 FR7). Resolution to a plain-text repo command is closed
    # under this build's own decision: if the working directory isn't a
    # git repo, or has zero commits, this is treated as a fatal
    # config-load failure -- consistent with every other "fail before
    # spending budget" decision in this system, rather than silently
    # falling back to an empty string that would make git_commit_hash
    # useless for reproducing which code version produced a run.
    raw.pop("git_commit_hash", None)
    try:
        raw["git_commit_hash"] = _resolve_git_commit_hash()
    except ConfigLoadError as e:
        # Re-wrap so the error carries the caller's real config path.
        # _resolve_git_commit_hash() has no idea which file is being loaded
        # and raises with a placeholder path, so letting it propagate
        # unchanged tells the user their config is invalid without telling
        # them which one.
        raise ConfigLoadError(path, e.errors) from e

    # Pydantic errors and filesystem-asset errors are accumulated together,
    # not fail-fast on whichever stage runs first: Feature 1's own
    # Acceptance Criteria require "a config with two independent errors
    # (e.g., bad alpha and missing persona pool)" to produce ONE
    # ConfigLoadError listing both. This deliberately deviates from
    # full_design_doc.md §3.1's sample implementation, which raises
    # immediately on the first Pydantic ValidationError and never reaches
    # the filesystem check at all -- that sample code cannot actually
    # satisfy this feature's own stated acceptance criterion, so the
    # acceptance criterion (the higher-authority contract) is what's
    # implemented here. Asset existence is checked directly against the
    # raw dict so it runs regardless of whether Pydantic validation itself
    # succeeded.
    errors: list[str] = []
    run: ExperimentRun | None = None
    try:
        run = ExperimentRun(**raw)
    except ValidationError as e:
        errors.extend(_format_validation_errors(e))

    errors.extend(_referenced_asset_errors(raw))

    if errors:
        raise ConfigLoadError(path, errors)

    assert run is not None
    return run


def _format_validation_errors(e: ValidationError) -> list[str]:
    """Renders Pydantic errors as "field: message" rather than raw dicts.

    str(err) on a Pydantic error dict prints the whole structure, including
    the full input payload repeated once per error. A config missing a dozen
    required fields produced a wall of text with the actual problems buried
    in it, which is the opposite of what a config error should do.
    """
    out: list[str] = []
    for err in e.errors():
        field = ".".join(str(p) for p in err["loc"]) or "<root>"
        message = err["msg"]
        if err["type"] == "extra_forbidden":
            message = "unknown field (check for a typo; unknown keys are rejected)"
        out.append(f"{field}: {message}")
    return out


def _bank_errors(raw: dict) -> list[str]:
    """Validates a bank-regime run's argument bank at config-load time.

    The bank is loaded here in full, not merely stat()ed, and every problem
    it reports is folded into this config's error list. A bank with an empty
    Hindi cell is a config error in exactly the sense the rest of this module
    means: the run would start, complete, cost its GPU hours, and produce a
    Hindi arm in which one argument was silently blank -- which is
    indistinguishable in the output from an argument that failed to persuade.
    Feature 1's whole premise is that such a thing is caught before anything
    is spent.

    Reads `regime` from the raw dict for the same reason the rest of this
    function does: it has to run whether or not Pydantic validation
    succeeded, so a config with both a bad alpha and a broken bank reports
    both.
    """
    if raw.get("regime") != "bank":
        return []

    bank_path = raw.get("bank_path")
    if not isinstance(bank_path, str) or not bank_path.strip():
        # ExperimentRun's own validator reports the missing field; repeating
        # it here would put the same problem in the list twice.
        return []

    path = Path(bank_path)
    if not path.exists():
        return [f"bank_path '{bank_path}' not found at {path}"]

    # Imported here rather than at module scope: config_loader is Tier 0 and
    # bank_manager is Tier 1, so a module-level import would invert the
    # dependency direction the whole architecture is built on.
    from sandbox.bank_manager import BankManager, BankValidationError

    try:
        BankManager(path)
    except BankValidationError as e:
        return [f"bank {bank_path}: {problem}" for problem in e.errors]
    return []


def _neutral_bank_errors(raw: dict) -> list[str]:
    """Validates a delivery-off control's neutral bank at config-load time.

    Loaded in full, for _bank_errors' reason: an empty Hindi cell here gives
    that arm's control a blank post every round, and a blank post is
    indistinguishable in the output from a message nobody moved for. The
    control is what every treatment effect is measured against, so a quietly
    broken one invalidates the arms it is the baseline for, not just itself.

    Reads regime and delivery from the raw dict so this runs whether or not
    Pydantic validation succeeded.
    """
    if raw.get("regime") != "bank" or raw.get("delivery") != "off":
        return []

    neutral_bank_path = raw.get("neutral_bank_path")
    if not isinstance(neutral_bank_path, str) or not neutral_bank_path.strip():
        # ExperimentRun's own validator reports the missing field; repeating
        # it here would put the same problem in the list twice.
        return []

    path = Path(neutral_bank_path)
    if not path.exists():
        return [f"neutral_bank_path '{neutral_bank_path}' not found at {path}"]

    # Imported inside the function for _bank_errors' reason: config_loader is
    # Tier 0 and bank_manager is Tier 1.
    from sandbox.bank_manager import BankValidationError, NeutralBank

    try:
        NeutralBank(path)
    except BankValidationError as e:
        return [f"neutral bank {neutral_bank_path}: {problem}" for problem in e.errors]
    return []

def _resolve_git_commit_hash() -> str:


    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ConfigLoadError(
            Path("<config>"),
            [f"could not resolve git_commit_hash (not a git repo, or no commits yet): {e}"],
        ) from e
    return result.stdout.strip()


def _referenced_asset_errors(raw: dict) -> list[str]:
    """
    Checks that meme_pool_id (if meme_injection.enabled) resolves to an
    actual dataset file, that the persona pool referenced by the run
    exists, that a bank-regime run's bank loads and validates, and that a
    delivery-off control's neutral bank does too. Returns ALL missing-asset
    errors found, not just the first.

    Reads directly from the raw (possibly Pydantic-invalid) dict rather
    than a validated ExperimentRun, so this check runs independently of
    whether schema validation itself succeeded -- see the accumulation
    note in load_run_config().
    """
    errors: list[str] = []
    errors.extend(_bank_errors(raw))
    errors.extend(_neutral_bank_errors(raw))

    meme_injection = raw.get("meme_injection")
    if isinstance(meme_injection, dict) and meme_injection.get("enabled") is True:
        meme_pool_id = meme_injection.get("meme_pool_id")
        if isinstance(meme_pool_id, str) and meme_pool_id:
            pool_path = Path("data/memes") / f"{meme_pool_id}.jsonl"
            if not pool_path.exists():
                errors.append(f"meme_pool_id '{meme_pool_id}' not found at {pool_path}")

    persona_pool_id = raw.get("persona_pool_id")
    if isinstance(persona_pool_id, str) and persona_pool_id:
        persona_path = Path("data/personas") / f"{persona_pool_id}.jsonl"
        if not persona_path.exists():
            errors.append(f"persona pool '{persona_pool_id}' not found at {persona_path}")

    return errors
