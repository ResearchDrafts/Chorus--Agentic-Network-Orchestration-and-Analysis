from unittest.mock import create_autospec

import pytest
import yaml

from sandbox.model_gateway import ModelGateway
from tests.factories import make_backend_response, make_logprob_response


def _base_config_dict() -> dict:
    return {
        "run_id": "test_run_0001",
        "rq_target": "RQ1_RQ2",
        "topic": "a neutral test topic",
        "alpha": 0.5,
        "M": 10,
        "N": 3,
        "K": 2,
        "trial_number": 1,
        "language_condition": "english",
        "model_backend_id": "gpt-4o",
        "stance_scale": [1, 2, 3, 4, 5, 6, 7],
        "persona_pool_id": "test_pool",
        "seed": 42,
        "temperature": 0.7,
    }


@pytest.fixture
def valid_config_dict() -> dict:
    return _base_config_dict()


@pytest.fixture
def write_config(tmp_path):
    """Writes a dict as a YAML file under tmp_path and returns its Path."""

    def _write(config: dict, filename: str = "run.yaml"):
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


@pytest.fixture
def mock_gateway():
    """A stand-in ModelGateway returning a well-formed stance reply.

    autospec so a signature change in ModelGateway breaks these tests rather
    than silently passing. Note generate() is NOT reassigned to a bare
    AsyncMock: create_autospec already produces an AsyncMock for async
    methods, and overwriting it would throw away exactly the signature
    enforcement autospec exists to provide. Only the return value is set.
    (tests/test_factories.py pins this behaviour, since getting it wrong is
    silent: the fixture still works, it just stops catching drift.)

    supports_vision and model_backend_id are set explicitly because both are
    plain instance attributes assigned in __init__, which autospec cannot see
    from the class alone.

    Tests drive alternate paths through generate.return_value / .side_effect:
        mock_gateway.generate.return_value = make_backend_response(text="junk")
    to trigger StanceParseFailure, or a side_effect list to script a
    retry-then-succeed sequence.
    """
    gateway = create_autospec(ModelGateway, instance=True)
    gateway.generate.return_value = make_backend_response()
    # The bank regime's second call. autospec already produced an AsyncMock
    # for it, for the same reason it did for generate(), so only the return
    # value is set here.
    #
    # The default read is all mass on 5, so probe_mean is 5.0 for every agent
    # on every round. Constant on purpose: the orchestrator assigns sides from
    # probe_mean, so a constant read gives every test a known, identical side
    # assignment in every language arm, which is what the matched-set
    # comparison needs. Tests that want movement or a side split override:
    #     mock_gateway.score_next_token.return_value = make_logprob_response(mean=2.0)
    #     mock_gateway.score_next_token.side_effect = [...]
    gateway.score_next_token.return_value = make_logprob_response(mean=5.0)
    gateway.model_backend_id = "gpt-4o"
    gateway.supports_vision = True
    return gateway


@pytest.fixture
def text_only_gateway(mock_gateway):
    """Same, but text-only, for exercising the vision-fallback path."""
    mock_gateway.supports_vision = False
    mock_gateway.model_backend_id = "gpt-3.5-turbo"
    return mock_gateway


@pytest.fixture
def bank_gateway(mock_gateway):
    """A gateway whose replies look like bank-regime replies.

    mock_gateway's default reply is "STANCE: 5.0\nmy neighbours shifted me",
    which is the shape the generated regime's parser needs. A bank reply is
    parsed for nothing and is simply prose, and the difference is not
    cosmetic: the orchestrator writes the reply verbatim into the agent's
    memory, so a mock emitting a STANCE line puts the literal word STANCE into
    later prompts and makes the "no prompt mentions STANCE" audit fail against
    the fixture rather than against the production code.

    The probe read stays mock_gateway's constant 5.0.
    """
    mock_gateway.generate.return_value = make_backend_response(
        text="The cost point is the one that lands for me."
    )
    return mock_gateway
