
import pytest

from sandbox.config_loader import ConfigLoadError, load_run_config


def test_valid_config_produces_experiment_run_with_git_hash(valid_config_dict, write_config):
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.run_id == "test_run_0001"
    assert run.git_commit_hash  # populated automatically, never from YAML
    assert len(run.git_commit_hash) == 40  # full sha


def test_git_commit_hash_in_yaml_is_ignored(valid_config_dict, write_config):
    valid_config_dict["git_commit_hash"] = "deadbeef" * 5
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.git_commit_hash != "deadbeef" * 5


def test_missing_file_raises_config_load_error(tmp_path):
    with pytest.raises(ConfigLoadError):
        load_run_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_load_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("topic: [unclosed")
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_missing_required_field_names_it(valid_config_dict, write_config):
    del valid_config_dict["topic"]
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("topic" in e for e in exc_info.value.errors)


def test_n_greater_or_equal_m_rejected(valid_config_dict, write_config):
    valid_config_dict["M"] = 5
    valid_config_dict["N"] = 5
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_meme_enabled_without_pool_id_rejected_before_filesystem_check(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True}
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


def test_meme_enabled_with_nonexistent_pool_rejected_naming_path(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True, "meme_pool_id": "does_not_exist_pool"}
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("data/memes" in e for e in exc_info.value.errors)


def test_meme_enabled_with_real_pool_succeeds(valid_config_dict, write_config):
    valid_config_dict["meme_injection"] = {"enabled": True, "meme_pool_id": "test_pool"}
    path = write_config(valid_config_dict)
    run = load_run_config(path)
    assert run.meme_injection.enabled is True


def test_two_independent_errors_both_surface(valid_config_dict, write_config):
    valid_config_dict["alpha"] = 1.5
    valid_config_dict["persona_pool_id"] = "no_such_pool"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    # alpha is a Pydantic-level failure (surfaces first, blocking the
    # filesystem check per the fail-before-filesystem-check ordering);
    # this asserts the error message names alpha specifically.
    assert any("alpha" in e for e in exc_info.value.errors)


def test_persona_pool_missing_alone_is_reported(valid_config_dict, write_config):
    valid_config_dict["persona_pool_id"] = "no_such_pool"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError) as exc_info:
        load_run_config(path)
    assert any("no_such_pool" in e for e in exc_info.value.errors)


def test_mode_other_than_multi_turn_rejected(valid_config_dict, write_config):
    valid_config_dict["mode"] = "single_exposure"
    path = write_config(valid_config_dict)
    with pytest.raises(ConfigLoadError):
        load_run_config(path)


# --- phase4.md D4: non-mapping YAML must still raise ConfigLoadError -----


@pytest.mark.parametrize(
    "yaml_text, described_as",
    [
        ("- a\n- b\n", "list"),
        ("just_a_string\n", "str"),
        ("42\n", "int"),
        ("true\n", "bool"),
    ],
)
def test_non_mapping_yaml_raises_config_load_error(tmp_path, yaml_text, described_as):
    """A syntactically valid YAML document need not be a mapping. Without the
    isinstance guard these raised raw TypeError/AttributeError out of
    raw.pop(), breaking load_run_config's documented contract that every
    failure surfaces as ConfigLoadError."""
    path = tmp_path / "run.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert "must be a mapping" in str(excinfo.value)
    assert described_as in str(excinfo.value)


# --- phase4.md D5: errors carry the caller's real config path ------------


def test_config_load_error_carries_the_real_path(tmp_path):
    path = tmp_path / "my_experiment.yaml"
    path.write_text("42\n")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert excinfo.value.path == path
    assert "my_experiment.yaml" in str(excinfo.value)


def test_missing_file_error_carries_the_real_path(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    assert excinfo.value.path == path


# --- readable validation errors -----------------------------------------


def test_validation_errors_are_field_message_not_raw_dicts(tmp_path):
    """str() on a Pydantic error dict prints the whole structure including the
    full input payload, repeated once per error, so a config missing a dozen
    fields buried the actual problems in a wall of text."""
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: x\ntopic: t\n")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    message = str(excinfo.value)
    assert "alpha: Field required" in message
    assert "'type':" not in message and "'loc':" not in message  # no raw dicts
    assert "errors.pydantic.dev" not in message  # no doc URLs


def test_unknown_key_error_names_the_field_and_explains(tmp_path, valid_config_dict, write_config):
    valid_config_dict["meme_injections"] = {"enabled": True}
    path = write_config(valid_config_dict)

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)

    message = str(excinfo.value)
    assert "meme_injections: unknown field" in message
    assert "typo" in message


# --- RQ1: the bank regime's referenced assets ----------------------------


def _bank_config(valid_config_dict, **overrides):
    valid_config_dict.update(
        regime="bank",
        bank_path="data/bank/test_bank.csv",
        probe_question="Should city bus fares stay subsidised?",
    )
    valid_config_dict.update(overrides)
    return valid_config_dict


HEADER = "item_id,arg_id,strength,side,english,hinglish,romanized_hindi,hindi"


def test_a_valid_bank_config_loads(valid_config_dict, write_config):
    run = load_run_config(write_config(_bank_config(valid_config_dict)))
    assert run.regime == "bank"
    assert run.bank_path == "data/bank/test_bank.csv"


def test_a_missing_bank_file_is_reported_with_its_path(valid_config_dict, write_config):
    path = write_config(_bank_config(valid_config_dict, bank_path="data/bank/nope.csv"))
    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)
    assert any("data/bank/nope.csv" in e for e in excinfo.value.errors)


def test_a_bank_that_fails_validation_is_rejected_at_config_load(
    tmp_path, valid_config_dict, write_config
):
    """The bank is loaded in full, not merely stat()ed.

    A bank with one empty Hindi cell would otherwise start, complete, cost its
    GPU hours, and produce a Hindi arm in which one argument was silently
    blank -- indistinguishable in the output from an argument that failed to
    persuade. Feature 1's premise is that this is caught before anything is
    spent.
    """
    bank = tmp_path / "broken.csv"
    bank.write_text(
        HEADER + "\n"
        "b1,a1,strong,pro,english text,hinglish text,roman text,\n"
        "b2,a2,weak,con,english text,hinglish text,roman text,hindi text\n"
    )
    path = write_config(_bank_config(valid_config_dict, bank_path=str(bank)))

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)
    assert any("empty hindi text" in e for e in excinfo.value.errors)


def test_a_one_sided_bank_is_rejected(tmp_path, valid_config_dict, write_config):
    bank = tmp_path / "one_sided.csv"
    bank.write_text(
        HEADER + "\n"
        "b1,a1,strong,pro,e,hi,rh,h\n"
        "b2,a2,weak,pro,e,hi,rh,h\n"
    )
    path = write_config(_bank_config(valid_config_dict, bank_path=str(bank)))

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)
    assert any("'con' side" in e for e in excinfo.value.errors)


def test_a_bank_error_and_a_schema_error_both_surface(
    tmp_path, valid_config_dict, write_config
):
    """Feature 1's accumulation contract, extended to the bank: a config with
    two independent problems takes one run to fix, not two."""
    bank = tmp_path / "broken.csv"
    bank.write_text(HEADER + "\n" + "b1,a1,strong,sideways,e,hi,rh,h\n")
    path = write_config(_bank_config(valid_config_dict, bank_path=str(bank), alpha=1.5))

    with pytest.raises(ConfigLoadError) as excinfo:
        load_run_config(path)
    errors = excinfo.value.errors
    assert any("alpha" in e for e in errors)
    assert any("side must be one of" in e for e in errors)


def test_bank_regime_without_a_bank_path_is_rejected(valid_config_dict, write_config):
    config = _bank_config(valid_config_dict)
    del config["bank_path"]
    with pytest.raises(ConfigLoadError, match="bank_path"):
        load_run_config(write_config(config))


def test_bank_regime_without_a_probe_question_is_rejected(valid_config_dict, write_config):
    """The one that would do the most damage silently: an English default
    would probe every language arm in English."""
    config = _bank_config(valid_config_dict)
    del config["probe_question"]
    with pytest.raises(ConfigLoadError, match="probe_question"):
        load_run_config(write_config(config))


def test_the_generated_regime_never_looks_for_a_bank(valid_config_dict, write_config):
    """Guards against over-correcting: a generated run with no bank_path must
    still load, and must not be slowed by a bank it does not use."""
    run = load_run_config(write_config(valid_config_dict))
    assert run.regime == "generated"
    assert run.bank_path is None


def test_a_graph_config_loads(valid_config_dict, write_config):
    valid_config_dict.update(
        M=8, N=3, neighbor_strategy="graph", graph_degree=4, graph_rewire_p=0.1
    )
    run = load_run_config(write_config(valid_config_dict))
    assert run.neighbor_strategy == "graph"
    assert run.graph_degree == 4


def test_a_graph_config_with_an_impossible_degree_is_rejected(
    valid_config_dict, write_config
):
    valid_config_dict.update(M=8, N=3, neighbor_strategy="graph", graph_degree=10)
    with pytest.raises(ConfigLoadError, match="less than M"):
        load_run_config(write_config(valid_config_dict))


# --- RQ1: the shipped configs --------------------------------------------


@pytest.mark.parametrize(
    "name", ["smoke", "smoke_hinglish", "smoke_off"]
)
def test_the_shipped_rq1_configs_load(name):
    """These are the only worked bank-regime examples a researcher has. If one
    drifts out of sync with the schema it is worse than no example."""
    from pathlib import Path

    run = load_run_config(Path(f"configs/rq1/{name}.yaml"))
    assert run.regime == "bank"
    assert run.neighbor_strategy == "graph"
    assert (run.M, run.N, run.K, run.graph_degree) == (8, 3, 3, 4)
    assert run.probe_question
    assert run.git_commit_hash


ARM_CONFIGS = (
    "smoke",
    "smoke_hinglish",
    "smoke_romanized_hindi",
    "smoke_hindi",
)


def test_all_four_arm_configs_load():
    """Every condition RQ1 manipulates ships as a worked example, including
    the two Roman-script ones, which are the contrast the experiment exists
    to measure and the easiest pair to conflate."""
    from pathlib import Path

    runs = {
        name: load_run_config(Path(f"configs/rq1/{name}.yaml")) for name in ARM_CONFIGS
    }
    assert {r.language_condition for r in runs.values()} == {
        "english", "hinglish", "romanized_hindi", "hindi"
    }
    assert len({r.run_id for r in runs.values()}) == 4


def test_the_rq1_configs_differ_in_exactly_three_fields():
    """The arms may differ in language_condition, probe_question and run_id,
    and in nothing else.

    run_id is bookkeeping (two arms must not write into one directory).
    language_condition selects the bank column and the directive line, and
    probe_question is the one piece of prompt text that is translated. The
    anchor labels are NOT: they are rendered inside English sentences -- the
    probe's scale line and the lean -- so translating them would code-mix the
    frame and leave the four arms holding four subtly different instruments.
    """
    from pathlib import Path

    runs = {
        name: load_run_config(Path(f"configs/rq1/{name}.yaml")).model_dump()
        for name in ARM_CONFIGS
    }
    baseline = runs["smoke"]
    for name, cfg in runs.items():
        differing = {k for k in baseline if baseline[k] != cfg[k]}
        assert differing <= {"run_id", "language_condition", "probe_question"}, name

    # And they really do differ in those: a config copied without editing the
    # question would probe that arm in another arm's language.
    assert len({cfg["probe_question"] for cfg in runs.values()}) == 4


def test_every_arm_carries_the_same_english_anchor_labels():
    from pathlib import Path

    labels = {
        (
            load_run_config(Path(f"configs/rq1/{name}.yaml")).stance_low_label,
            load_run_config(Path(f"configs/rq1/{name}.yaml")).stance_high_label,
        )
        for name in ARM_CONFIGS
    }
    assert len(labels) == 1
    low, high = labels.pop()
    assert not any("\u0900" <= ch <= "\u097f" for ch in low + high)


def test_the_delivery_off_config_differs_only_in_delivery():
    """And in run_id, so the two arms do not write into the same directory,
    and in neutral_message, which only the control delivers."""
    from pathlib import Path

    on = load_run_config(Path("configs/rq1/smoke.yaml")).model_dump()
    off = load_run_config(Path("configs/rq1/smoke_off.yaml")).model_dump()

    differing = {k for k in on if on[k] != off[k]}
    assert differing == {"delivery", "run_id", "neutral_message"}
    assert on["neutral_message"] is None
    assert off["neutral_message"].strip()


def test_delivery_is_quoted_in_every_shipped_config():
    """YAML 1.1 reads bare on/off as booleans, so `delivery: off` arrives as
    False and the schema rejects it. The quotes are load-bearing, and a
    well-meaning tidy-up would remove them."""
    import yaml
    from pathlib import Path

    for name in (*ARM_CONFIGS, "smoke_off"):
        raw = yaml.safe_load(Path(f"configs/rq1/{name}.yaml").read_text())
        assert isinstance(raw["delivery"], str), f"{name}: delivery parsed as {raw['delivery']!r}"
