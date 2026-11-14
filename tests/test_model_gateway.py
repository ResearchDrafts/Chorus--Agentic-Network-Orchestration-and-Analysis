from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

import litellm
import pytest

from sandbox.cost_tracker import CostTracker
from sandbox.model_gateway import FatalGatewayError, ModelGateway
from sandbox.rate_limiter import RateLimiter
from tests.factories import make_run


@pytest.fixture(autouse=True)
def no_real_sleep():
    """Tenacity's wait_exponential backoff (min=2s) would otherwise make
    every retry test take several real seconds; patching asyncio.sleep
    keeps retry tests fast and deterministic without touching the
    production retry configuration."""
    with patch("asyncio.sleep", new_callable=AsyncMock):
        yield


@pytest.fixture
def rate_limiter():
    limiter = create_autospec(RateLimiter, instance=True)
    limiter.acquire = AsyncMock()
    return limiter


@pytest.fixture
def cost_tracker():
    return create_autospec(CostTracker, instance=True)


# Local, not in factories.py: this is a raw litellm-shaped response, and it
# deliberately DOES define response_ms even though real ModelResponse objects
# do not (phase4.md D2). Keeping it lets the D2 tests below prove production
# code ignores the field rather than merely tolerating its absence.
def _make_raw_response(text="ok", prompt_tokens=10, completion_tokens=5, response_ms=123):
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        response_ms=response_ms,
    )
    raw.model_dump = lambda: {"text": text}
    return raw


async def test_transient_error_then_success_retries_and_succeeds(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    success = _make_raw_response(text="hello")
    mock_completion = AsyncMock(
        side_effect=[
            litellm.exceptions.RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4o"),
            success,
        ]
    )

    with patch("litellm.acompletion", mock_completion):
        response = await gateway.generate("prompt")

    assert response.text == "hello"
    assert mock_completion.call_count == 2


async def test_authentication_error_fails_on_first_attempt_no_retry(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
    )

    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.generate("prompt")

    assert mock_completion.call_count == 1


def test_unrecognized_model_id_raises_value_error_at_construction(rate_limiter, cost_tracker):
    with pytest.raises(ValueError):
        ModelGateway("totally-made-up-model-id", rate_limiter, cost_tracker)


async def test_text_call_records_cost_with_text_modality(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=None)

    cost_tracker.record.assert_called_once_with(
        provider="openai",
        model_id="gpt-4o",
        modality="text",
        prompt_tokens=10,
        completion_tokens=5,
    )


async def test_vision_call_records_cost_with_vision_modality(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=b"fake-image-bytes")

    cost_tracker.record.assert_called_once_with(
        provider="openai",
        model_id="gpt-4o",
        modality="vision",
        prompt_tokens=10,
        completion_tokens=5,
    )


async def test_rate_limiter_release_called_once_on_success(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")

    rate_limiter.release.assert_called_once_with("openai")


async def test_rate_limiter_release_called_once_even_when_call_raises(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
    )

    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.generate("prompt")

    rate_limiter.release.assert_called_once_with("openai")


# --- phase4.md D1: pricing lookup must use the BARE model id -------------


@pytest.mark.parametrize(
    "model_backend_id, expected_provider, expected_model_id",
    [
        ("gpt-4o", "openai", "gpt-4o"),
        ("gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("gpt-3.5-turbo", "openai", "gpt-3.5-turbo"),
        ("anthropic/claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("google/gemini-2.0-flash", "google", "gemini-2.0-flash"),
        ("local/llama-3.1-70b", "local", "llama-3.1-70b"),
        ("local/llama-3.1-8b", "local", "llama-3.1-8b"),
    ],
)
async def test_cost_record_uses_bare_model_id(
    rate_limiter, cost_tracker, model_backend_id, expected_provider, expected_model_id
):
    """Previously generate() passed the full prefixed model_backend_id as
    model_id while provider was already stripped, so the pricing lookup became
    pricing["anthropic"]["anthropic/claude-sonnet-4-6"] and raised for every
    model except a bare OpenAI one."""
    gateway = ModelGateway(model_backend_id, rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")

    kwargs = cost_tracker.record.call_args.kwargs
    assert kwargs["provider"] == expected_provider
    assert kwargs["model_id"] == expected_model_id


@pytest.mark.parametrize(
    "model_backend_id",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-2.0-flash",
        "local/llama-3.1-70b",
        "local/llama-3.1-8b",
    ],
)
async def test_every_priced_model_resolves_against_real_cost_tracker(
    rate_limiter, model_backend_id
):
    """End-to-end against the real pricing_table.yaml and a real CostTracker,
    not an autospec mock. This is the seam that was completely untested and
    that D1 broke: 6 of 7 priced models raised ValueError at record() time,
    after the API call had already been paid for."""
    run = make_run(model_backend_id=model_backend_id)
    tracker = CostTracker(run, Path("pricing_table.yaml"))
    gateway = ModelGateway(model_backend_id, rate_limiter, tracker)

    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt")  # must not raise

    assert tracker.total_usd >= 0.0


# --- phase4.md D2: latency is measured, not read off the response --------


async def test_latency_is_measured_not_read_from_provider_response(
    rate_limiter, cost_tracker
):
    """full_design_doc.md Sec 3.5 reads raw.response_ms, which does not exist
    on litellm's ModelResponse. The mock below supplies response_ms=999 to
    prove production code no longer reads it."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch(
        "litellm.acompletion",
        AsyncMock(return_value=_make_raw_response(response_ms=999)),
    ):
        response = await gateway.generate("prompt")

    assert response.latency_ms != 999
    assert isinstance(response.latency_ms, int)
    assert response.latency_ms >= 0


async def test_latency_survives_a_response_object_without_response_ms(
    rate_limiter, cost_tracker
):
    """The real regression: a response lacking response_ms entirely, as every
    real litellm ModelResponse does. This raised AttributeError, which is
    neither gateway error type and so aborted the whole run."""
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    raw.model_dump = lambda: {"text": "ok"}

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=raw)):
        response = await gateway.generate("prompt")

    assert isinstance(response.latency_ms, int)


# --- audit B1/B3: reaching a self-hosted server -------------------------


@pytest.mark.parametrize(
    "model_backend_id, expect_vision",
    [
        ("hosted_vllm/Qwen/Qwen2.5-7B-Instruct", False),
        ("hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct", True),
        ("hosted_vllm/Qwen/Qwen2.5-VL-3B-Instruct", True),
        ("groq/openai/gpt-oss-120b", False),
        ("cerebras/glm-4.7", False),
    ],
)
def test_self_hosted_and_free_tier_models_construct(
    rate_limiter, cost_tracker, model_backend_id, expect_vision
):
    """_resolve_vision_support raises on an unrecognized id at construction,
    so an unlisted model cannot start a run at all."""
    gateway = ModelGateway(model_backend_id, rate_limiter, cost_tracker)
    assert gateway.supports_vision is expect_vision


async def test_api_base_is_forwarded_to_litellm(rate_limiter, cost_tracker):
    """Without this there is no way to reach a local vLLM server: litellm
    routes on the model prefix alone and would call a hosted provider."""
    gateway = ModelGateway(
        "hosted_vllm/Qwen/Qwen2.5-7B-Instruct", rate_limiter, cost_tracker,
        api_base="http://localhost:8000/v1",
    )
    mock_completion = AsyncMock(return_value=_make_raw_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("prompt")

    assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:8000/v1"


async def test_api_base_defaults_to_none_for_hosted_providers(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(return_value=_make_raw_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("prompt")

    assert mock_completion.call_args.kwargs["api_base"] is None


@pytest.mark.parametrize(
    "model_backend_id, bare, modality",
    [
        ("hosted_vllm/Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "text"),
        ("hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "vision"),
        ("groq/openai/gpt-oss-120b", "gpt-oss-120b", "text"),
        ("cerebras/gpt-oss-120b", "gpt-oss-120b", "text"),
    ],
)
async def test_every_configured_backend_has_a_pricing_entry(
    rate_limiter, model_backend_id, bare, modality
):
    """CostTracker raises on a missing entry AFTER the call completes, so a
    gap here kills a run having already spent the compute."""
    run = make_run(model_backend_id=model_backend_id)
    tracker = CostTracker(run, Path("pricing_table.yaml"))
    gateway = ModelGateway(model_backend_id, rate_limiter, tracker)

    image = b"fake" if modality == "vision" else None
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        await gateway.generate("prompt", image=image)  # must not raise

    assert tracker.total_usd == 0.0  # genuinely free, not merely unpriced


# --- concurrency cap: bounding in-flight requests ------------------------
#
# These use asyncio.Barrier rather than sleeps to force genuine overlap: the
# module's autouse no_real_sleep fixture patches asyncio.sleep, so a mock that
# awaits it never actually yields and tasks cannot interleave.


async def _peak_concurrency(gateway, n_requests, barrier_size):
    """Runs n_requests through the gateway, returning peak simultaneous calls.

    Each mocked call waits on a barrier of barrier_size, so it cannot return
    until that many are genuinely in flight at once.
    """
    import asyncio

    barrier = asyncio.Barrier(barrier_size)
    in_flight = peak = 0

    async def tracked(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        in_flight -= 1
        return _make_raw_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=tracked)):
        await asyncio.gather(*(gateway.generate("p") for _ in range(n_requests)))
    return peak


async def test_max_concurrency_bounds_simultaneous_requests(rate_limiter, cost_tracker):
    """The Orchestrator dispatches all M agents at once. Against a local
    server batching 32 at a time, M=100 leaves 68 connections open and idle
    each turn; they accumulate until the server closes them and the run dies
    with "Server disconnected" while the process is still healthy."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    peak = await _peak_concurrency(gateway, n_requests=40, barrier_size=4)
    assert peak == 4, f"expected exactly the cap, got {peak}"


async def test_unbounded_by_default(rate_limiter, cost_tracker):
    """Hosted providers pool connections themselves, so omitting the cap
    preserves the previous behaviour."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    peak = await _peak_concurrency(gateway, n_requests=20, barrier_size=20)
    assert peak == 20, f"expected all 20 in flight, got {peak}"


async def test_all_requests_still_complete_under_the_cap(rate_limiter, cost_tracker):
    """Bounding is not dropping: every request runs, just not all at once."""
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_raw_response())):
        results = await asyncio.gather(*(gateway.generate("p") for _ in range(40)))
    assert len(results) == 40
    assert all(r.text == "ok" for r in results)


async def test_cap_holds_across_retries(rate_limiter, cost_tracker):
    """The slot is held for the whole call, retries included, so a retrying
    agent never adds a connection beyond the cap."""
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=2)
    in_flight = peak = 0
    calls = {"n": 0}

    async def flaky(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        calls["n"] += 1
        should_fail = calls["n"] % 3 == 1
        in_flight -= 1
        if should_fail:
            raise litellm.exceptions.RateLimitError(
                message="429", llm_provider="openai", model="gpt-4o")
        return _make_raw_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=flaky)):
        await asyncio.gather(*(gateway.generate("p") for _ in range(12)))

    assert peak <= 2, f"cap of 2 exceeded during retries: {peak}"
    assert calls["n"] > 12, "expected retries to have occurred"


# --- RQ1: score_next_token, the probe call -------------------------------


def _logprob_entry(token, logprob):
    """A litellm ChatCompletionTokenLogprob-shaped object."""
    return SimpleNamespace(token=token, logprob=logprob)


def _make_probe_response(
    entries=(("5", -0.1), (" 6", -2.3), (" the", -5.0)),
    prompt_tokens=8,
    completion_tokens=1,
    sampled="5",
    logprobs=True,
):
    """A litellm-shaped response carrying next-token logprobs.

    `logprobs=False` produces the response a server that does not support
    logprobs returns: everything else present, that field absent.
    """
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=sampled),
                logprobs=(
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token=sampled,
                                logprob=entries[0][1],
                                top_logprobs=[_logprob_entry(t, lp) for t, lp in entries],
                            )
                        ]
                    )
                    if logprobs
                    else None
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    raw.model_dump = lambda: {"probe": True}
    return raw


async def test_score_next_token_parses_the_logprob_pairs(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_probe_response())):
        response = await gateway.score_next_token("probe prompt")

    assert response.top_logprobs == [("5", -0.1), (" 6", -2.3), (" the", -5.0)]
    assert response.text == "5"
    assert response.prompt_tokens == 8
    assert response.completion_tokens == 1
    assert isinstance(response.latency_ms, int) and response.latency_ms >= 0


async def test_the_parsed_pairs_feed_straight_into_the_scorer(rate_limiter, cost_tracker):
    """Closes the loop between the two modules: the shape the gateway produces
    is the shape probe_scorer consumes. Asserting the parse alone would not
    catch a drift between them."""
    from sandbox.probe_scorer import score

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_probe_response())):
        response = await gateway.score_next_token("probe prompt")

    result = score(response.top_logprobs, mass_floor=0.5)
    assert 5.0 <= result.mean <= 6.0  # mass on 5 and " 6", read as the same scale


async def test_probe_parameters_are_fixed_at_one_token_and_zero_temperature(
    rate_limiter, cost_tracker
):
    """The probe is a measurement, not a generation: sampling temperature
    would add noise to the thing being measured, and a second token would be
    read by nothing."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, top_logprobs=12)
    mock_completion = AsyncMock(return_value=_make_probe_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.score_next_token("probe prompt")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 1
    assert kwargs["temperature"] == 0.0
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 12


async def test_the_top_logprobs_count_comes_from_construction(rate_limiter, cost_tracker):
    """All seven digits must be reachable within this many, so a run probed at
    5 is not comparable to one probed at 20 -- which is why the value lives on
    the run and is recorded in run_config.json."""
    for requested in (1, 5, 20, 50):
        gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, top_logprobs=requested)
        mock_completion = AsyncMock(return_value=_make_probe_response())
        with patch("litellm.acompletion", mock_completion):
            await gateway.score_next_token("p")
        assert mock_completion.call_args.kwargs["top_logprobs"] == requested


def test_an_invalid_top_logprobs_count_raises_at_construction(rate_limiter, cost_tracker):
    with pytest.raises(ValueError, match="top_logprobs"):
        ModelGateway("gpt-4o", rate_limiter, cost_tracker, top_logprobs=0)


async def test_api_base_is_forwarded_on_the_probe_path_too(rate_limiter, cost_tracker):
    """The probe talks to the same self-hosted server the replies do. Without
    this the reply path would reach localhost and the probe path would not."""
    gateway = ModelGateway(
        "hosted_vllm/Qwen/Qwen2.5-7B-Instruct", rate_limiter, cost_tracker,
        api_base="http://localhost:8000/v1",
    )
    mock_completion = AsyncMock(return_value=_make_probe_response())
    with patch("litellm.acompletion", mock_completion):
        await gateway.score_next_token("p")

    assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:8000/v1"


async def test_a_probe_records_cost_as_text(rate_limiter, cost_tracker):
    """A probe is a billed request. The bank regime makes two calls per agent
    per round, so a probe path that recorded nothing would under-report a
    run's cost by about half."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_probe_response())):
        await gateway.score_next_token("p")

    cost_tracker.record.assert_called_once_with(
        provider="openai", model_id="gpt-4o", modality="text",
        prompt_tokens=8, completion_tokens=1,
    )


async def test_the_probe_releases_the_rate_limiter_on_success_and_on_failure(
    rate_limiter, cost_tracker
):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_probe_response())):
        await gateway.score_next_token("p")
    assert rate_limiter.release.call_count == 1

    failing = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o")
    )
    with patch("litellm.acompletion", failing):
        with pytest.raises(FatalGatewayError):
            await gateway.score_next_token("p")
    assert rate_limiter.release.call_count == 2


async def test_a_transient_error_on_the_probe_path_retries(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=[
            litellm.exceptions.RateLimitError(
                message="429", llm_provider="openai", model="gpt-4o"),
            _make_probe_response(),
        ]
    )
    with patch("litellm.acompletion", mock_completion):
        response = await gateway.score_next_token("p")

    assert mock_completion.call_count == 2
    assert response.top_logprobs


async def test_an_auth_error_on_the_probe_path_is_fatal_and_not_retried(
    rate_limiter, cost_tracker
):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o")
    )
    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.score_next_token("p")

    assert mock_completion.call_count == 1


async def test_a_server_that_does_not_support_logprobs_is_fatal_not_retried(
    rate_limiter, cost_tracker
):
    """BadRequestError covers a server without logprobs support. Retrying
    cannot make a server grow the feature, and the run should stop rather than
    burn three attempts per agent per round discovering that."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.BadRequestError(
            message="logprobs is not supported", llm_provider="openai", model="gpt-4o")
    )
    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(FatalGatewayError):
            await gateway.score_next_token("p")

    assert mock_completion.call_count == 1


# --- a context-length overflow is fatal for the RUN ----------------------


@pytest.mark.parametrize("call", ["generate", "score_next_token"])
async def test_a_context_window_overflow_is_not_a_per_agent_failure(
    call, rate_limiter, cost_tracker
):
    """ContextLengthExceeded, not FatalGatewayError.

    The orchestrator absorbs FatalGatewayError per agent, which is right for
    an auth failure: one round is lost and the run continues. It is wrong
    here. A prompt overflows because that agent's history has grown, so the
    same agent overflows every round from then on, and the agents that
    overflow first are the ones whose language tokenizes least efficiently --
    so absorbing it would drop the Hinglish arm's most talkative agents and
    keep the English arm's.
    """
    from sandbox.model_gateway import ContextLengthExceeded

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=litellm.exceptions.ContextWindowExceededError(
            message="This model's maximum context length is 8192 tokens",
            llm_provider="openai",
            model="gpt-4o",
        )
    )
    with patch("litellm.acompletion", mock_completion):
        with pytest.raises(ContextLengthExceeded):
            await getattr(gateway, call)("p")

    # Not retried: the prompt will not get shorter on a second attempt.
    assert mock_completion.call_count == 1
    assert not issubclass(ContextLengthExceeded, FatalGatewayError)


# --- missing logprobs: raise, never return an empty list -----------------


@pytest.mark.parametrize(
    "raw_builder, detail",
    [
        (lambda: _make_probe_response(logprobs=False), "logprobs is absent"),
        (lambda: _probe_response_with(logprobs=SimpleNamespace(content=[])), "content is empty"),
        (lambda: _probe_response_with(logprobs=SimpleNamespace(content=None)), "content is empty"),
        (
            lambda: _probe_response_with(
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(token="5", logprob=-0.1, top_logprobs=[])]
                )
            ),
            "top_logprobs is empty",
        ),
        (lambda: _probe_response_with(choices=[]), "no choices"),
    ],
)
async def test_a_response_without_logprobs_raises_rather_than_returning_empty(
    rate_limiter, cost_tracker, raw_builder, detail
):
    """An empty list would reach probe_scorer and fail there -- but once per
    agent per round, which reads in the output like a model refusing to
    answer. The real cause is one wiring error and should surface as one loud
    failure, with the diagnosis in the message."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    with patch("litellm.acompletion", AsyncMock(return_value=raw_builder())):
        with pytest.raises(FatalGatewayError) as excinfo:
            await gateway.score_next_token("p")

    assert detail in str(excinfo.value)
    assert "logprobs" in str(excinfo.value)


def _probe_response_with(**overrides):
    raw = _make_probe_response()
    if "choices" in overrides:
        raw.choices = overrides["choices"]
    if "logprobs" in overrides:
        raw.choices[0].logprobs = overrides["logprobs"]
    return raw


async def test_an_entry_missing_its_token_or_logprob_raises(rate_limiter, cost_tracker):
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    raw = _probe_response_with(
        logprobs=SimpleNamespace(
            content=[
                SimpleNamespace(
                    token="5", logprob=-0.1,
                    top_logprobs=[SimpleNamespace(token="5", logprob=None)],
                )
            ]
        )
    )
    with patch("litellm.acompletion", AsyncMock(return_value=raw)):
        with pytest.raises(FatalGatewayError, match="missing token or logprob"):
            await gateway.score_next_token("p")


async def test_dict_shaped_logprobs_are_also_accepted(rate_limiter, cost_tracker):
    """Some providers' entries arrive as plain dicts once litellm has
    round-tripped them through JSON."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    raw = _probe_response_with(
        logprobs={
            "content": [
                {
                    "token": "4",
                    "logprob": -0.2,
                    "top_logprobs": [{"token": "4", "logprob": -0.2}, {"token": " 5", "logprob": -1.8}],
                }
            ]
        }
    )
    with patch("litellm.acompletion", AsyncMock(return_value=raw)):
        response = await gateway.score_next_token("p")

    assert response.top_logprobs == [("4", -0.2), (" 5", -1.8)]


# --- generate() is unchanged by any of the above -------------------------


async def test_generate_still_requests_no_logprobs(rate_limiter, cost_tracker):
    """A None top_logprobs on a generate() response means "this was not a
    probe", never "the probe came back empty"."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(return_value=_make_raw_response())
    with patch("litellm.acompletion", mock_completion):
        response = await gateway.generate("prompt")

    assert "logprobs" not in mock_completion.call_args.kwargs
    assert "max_tokens" not in mock_completion.call_args.kwargs
    assert response.top_logprobs is None


# --- the concurrency cap applies to scoring too --------------------------


async def _peak_probe_concurrency(gateway, n_requests, barrier_size):
    """As _peak_concurrency, but driving score_next_token."""
    import asyncio

    barrier = asyncio.Barrier(barrier_size)
    in_flight = peak = 0

    async def tracked(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        in_flight -= 1
        return _make_probe_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=tracked)):
        await asyncio.gather(*(gateway.score_next_token("p") for _ in range(n_requests)))
    return peak


async def test_max_concurrency_bounds_simultaneous_probes(rate_limiter, cost_tracker):
    """The bank regime makes two calls per agent per round, so a probe path
    outside the semaphore would double the in-flight count the cap exists to
    bound -- and the failure mode (a local server closing accumulated idle
    connections) looks like a healthy server dropping the run."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    peak = await _peak_probe_concurrency(gateway, n_requests=40, barrier_size=4)
    assert peak == 4, f"expected exactly the cap, got {peak}"


async def test_replies_and_probes_share_one_cap(rate_limiter, cost_tracker):
    """One semaphore, not one each: the server's batch size bounds total
    requests, not requests per kind."""
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=3)
    in_flight = peak = 0
    barrier = asyncio.Barrier(3)

    async def tracked(*a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        in_flight -= 1
        return _make_probe_response() if k.get("max_tokens") == 1 else _make_raw_response()

    with patch("litellm.acompletion", AsyncMock(side_effect=tracked)):
        await asyncio.gather(
            *(gateway.generate("p") for _ in range(10)),
            *(gateway.score_next_token("p") for _ in range(10)),
        )

    assert peak == 3, f"cap of 3 exceeded across both call types: {peak}"


async def test_all_probes_complete_under_the_cap(rate_limiter, cost_tracker):
    import asyncio

    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker, max_concurrency=4)
    with patch("litellm.acompletion", AsyncMock(return_value=_make_probe_response())):
        results = await asyncio.gather(*(gateway.score_next_token("p") for _ in range(40)))

    assert len(results) == 40
    assert all(r.top_logprobs for r in results)


# --- the reply seed -------------------------------------------------------


async def test_a_seed_reaches_the_provider(rate_limiter, cost_tracker):
    """The whole point: the server's sampler becomes a function of the run's
    seed, so a rerun reproduces the replies and therefore the probe readings
    taken from the history they build."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(return_value=_make_raw_response())

    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("p", seed=123)

    assert mock_completion.await_args.kwargs["seed"] == 123


async def test_no_seed_sends_no_seed_key(rate_limiter, cost_tracker):
    """Absent, not null. A generated-regime call must be byte-identical to
    what it was before the parameter existed, and a provider that treats a
    missing key and an explicit null differently would otherwise change that
    regime's traffic for a reason it never asked for."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(return_value=_make_raw_response())

    with patch("litellm.acompletion", mock_completion):
        await gateway.generate("p")

    assert "seed" not in mock_completion.await_args.kwargs


async def test_a_transient_retry_reuses_the_same_seed(rate_limiter, cost_tracker):
    """A retry is the same request, not a new one. Re-sampling on a 429 would
    hand back a different reply for a failure the agent had no part in, and a
    rerun that hit no 429 would then diverge from one that did."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    mock_completion = AsyncMock(
        side_effect=[
            litellm.exceptions.RateLimitError(
                message="rate limited", llm_provider="openai", model="gpt-4o"
            ),
            _make_raw_response(text="after the retry"),
        ]
    )

    with patch("litellm.acompletion", mock_completion):
        response = await gateway.generate("p", seed=777)

    assert response.text == "after the retry"
    assert mock_completion.await_count == 2
    assert [c.kwargs["seed"] for c in mock_completion.await_args_list] == [777, 777]


async def test_the_probe_call_is_never_seeded(rate_limiter, cost_tracker):
    """score_next_token runs at temperature 0 and samples nothing worth
    pinning; a seed there would suggest the measurement had a draw in it."""
    gateway = ModelGateway("gpt-4o", rate_limiter, cost_tracker)
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="5"),
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            top_logprobs=[SimpleNamespace(token="5", logprob=-0.1)]
                        )
                    ]
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1),
    )
    raw.model_dump = lambda: {}
    mock_completion = AsyncMock(return_value=raw)

    with patch("litellm.acompletion", mock_completion):
        await gateway.score_next_token("p")

    assert "seed" not in mock_completion.await_args.kwargs
