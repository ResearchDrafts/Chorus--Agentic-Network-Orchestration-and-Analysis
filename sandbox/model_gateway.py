"""Feature 7: Model Gateway.

The single normalized async interface to every LLM provider, backed by
LiteLLM. Direct port of full_design_doc.md Sec 3.5.

Per Phase 1's summary, CostTracker.record() is synchronous (not
`async def`), so generate() calls it as a plain call, never awaited.

Two entry points, not one. generate() is the original: free-text completion,
optionally multimodal. score_next_token() is the RQ1 probe: one token at
temperature 0 with the top logprobs attached, from which probe_scorer reads a
stance. They share the concurrency slot, the rate limiter, the retry policy
and the cost record deliberately -- a probe is a billed request against the
same server, and the bank regime makes two calls per agent per round, so a
probe path that bypassed the semaphore would double the in-flight count the
cap exists to bound.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from dataclasses import dataclass

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sandbox.cost_tracker import CostTracker
from sandbox.rate_limiter import RateLimiter

# Matches full_design_doc.md Sec 3.5's _resolve_vision_support table
# exactly. This is a strict superset of pricing_table.yaml's current
# entries -- llava and qwen-vl are recognized here as vision-capable per
# the HLD's mention of local vision models, even though pricing_table.yaml
# has no pricing entry for them yet. Using either of those two in a real
# run would surface loudly via CostTracker's "no pricing entry" ValueError
# at record() time, consistent with this system's fail-before-spending
# philosophy, rather than silently misclassifying them as unrecognized.
VISION_CAPABLE = {
    "gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "gemini-2.0-flash", "llava", "qwen-vl",
    # Self-hosted VLMs (phase5,6.md A.4). Serving one of these for a whole RQ3
    # campaign, text turns included, keeps meme and non-meme turns on the same
    # model, so they differ in content rather than in which model produced them.
    "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct",
}
TEXT_ONLY = {
    "gpt-3.5-turbo", "llama-3.1-8b", "llama-3.1-70b",
    # Self-hosted text models.
    "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
    # Hosted free-tier fallback (phase5,6.md A.3). None of these accepts
    # images: putting one in VISION_CAPABLE would send image payloads a
    # text-only endpoint rejects, and a meme-enabled run would die mid-turn.
    "gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "glm-4.7",
    # Ollama tags, for local smoke tests on hardware vLLM cannot target
    # (Apple Silicon). Ollama keeps the ":tag" suffix in the model name, so
    # the bare id is "qwen2.5:7b", not "qwen2.5".
    "qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b", "llama3.1:8b",
}


class TransientGatewayError(Exception):
    """Wraps any provider error judged retryable (timeouts, 429s, 5xxs)."""


class FatalGatewayError(Exception):
    """Wraps any provider error judged non-retryable (auth failure,
    invalid model id, content policy rejection) -- never retried."""


class ContextLengthExceeded(Exception):
    """The prompt did not fit in the model's context window.

    Deliberately NOT a FatalGatewayError. The orchestrator absorbs that one
    per agent, which is right for an auth failure or a content rejection --
    one agent's round is lost and the run continues. It is wrong here. A
    prompt that overflows does so because the agent's history has grown, so
    the same agent overflows again every round from then on, and the agents
    that overflow first are the ones whose language tokenizes least
    efficiently. A run that absorbed this would quietly drop the Hinglish
    arm's most talkative agents and keep the English arm's, which is a
    language-correlated loss in exactly the comparison RQ1 reports.

    So it ends the run. The remedy is a smaller K, a window
    (memory_rounds), or a model with a longer context -- all decisions for
    the person running it, not for the loop.
    """


@dataclass
class BackendResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_provider_response: dict
    # Populated only by score_next_token: [(token_text, logprob), ...] for the
    # single token the model was asked for. None from generate(), which does
    # not request logprobs at all -- so a None here means "this was not a
    # probe", never "the probe came back empty". An empty probe raises instead.
    top_logprobs: list[tuple[str, float]] | None = None


class ModelGateway:
    def __init__(
        self,
        model_backend_id: str,
        rate_limiter: RateLimiter,
        cost_tracker: CostTracker,
        api_base: str | None = None,
        max_concurrency: int | None = None,
        top_logprobs: int = 20,
    ):
        self.model_backend_id = model_backend_id
        self.supports_vision = _resolve_vision_support(model_backend_id)
        self._rate_limiter = rate_limiter
        self._cost_tracker = cost_tracker
        # Caps simultaneous in-flight requests. The Orchestrator dispatches all
        # M agents at once, so without this a turn opens M connections and any
        # beyond the server's batch size sit open waiting. Against a local vLLM
        # with --max-num-seqs 32, M=100 leaves 68 connections idle per turn;
        # they accumulate across turns until the server starts closing them and
        # the run dies with "Server disconnected" while the server process is
        # still perfectly healthy.
        #
        # This is NOT what RateLimiter does. That enforces a minimum interval
        # between requests and serializes a provider to one at a time, which is
        # right for a metered hosted API and ruinous for a batching local
        # server. A semaphore keeps many requests in flight, just a bounded
        # number, which is exactly what continuous batching wants.
        #
        # Set it to the server's batch size. None means unbounded, preserving
        # the previous behaviour for hosted providers that pool connections
        # themselves.
        self._semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )
        # Endpoint for a self-hosted backend, e.g. "http://localhost:8000/v1"
        # for vLLM. Without this there is no way to reach a local server at
        # all: litellm routes by the model prefix alone and would try to call
        # a hosted provider. None keeps hosted-provider behaviour unchanged.
        self._api_base = api_base
        # How many top logprobs score_next_token asks for. Set at
        # construction, from ExperimentRun.probe_top_logprobs, so the value
        # that produced a dataset is recorded in run_config.json: all seven
        # digits have to be reachable within this many, and a run probed at 5
        # is not comparable to one probed at 20.
        if top_logprobs < 1:
            raise ValueError(f"top_logprobs must be >= 1, got {top_logprobs}")
        self._top_logprobs = top_logprobs

    async def generate(
        self,
        prompt_text: str,
        image: bytes | None = None,
        temperature: float = 0.7,
    ) -> BackendResponse:
        """
        Normalizes a call to self.model_backend_id via LiteLLM. If image
        is provided but self.supports_vision is False, the CALLER is
        responsible for having already substituted a caption-only prompt
        (via vision_fallback.py) BEFORE calling this method -- this method
        does not itself perform the fallback substitution, it only
        reports supports_vision so callers can decide.
        """
        provider = self._provider_name()
        # The semaphore is held across the whole call, retries included, so an
        # agent retrying a StanceParseFailure does not add a connection beyond
        # the cap.
        async with self._concurrency_slot():
            await self._rate_limiter.acquire(provider)
            try:
                response = await self._call_with_retry(prompt_text, image, temperature)
            finally:
                self._rate_limiter.release(provider)

        modality = "vision" if image is not None else "text"
        self._cost_tracker.record(
            provider=provider,
            # Bare model id, not the full model_backend_id. pricing_table.yaml
            # nests as provider -> model_id, so passing the prefixed form here
            # while provider is already stripped looks up
            # pricing["anthropic"]["anthropic/claude-sonnet-4-6"] and fails for
            # every model except a bare OpenAI one. Normalized the same way
            # _resolve_vision_support() does.
            model_id=_bare_model_id(self.model_backend_id),
            modality=modality,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return response

    async def score_next_token(self, prompt_text: str) -> BackendResponse:
        """One token at temperature 0, with the top logprobs attached.

        The probe call. Deliberately the same shape as generate() -- same
        concurrency slot, same rate limiter, same retry policy, same cost
        record -- because a probe is a billed request against the same server,
        and a probe path that bypassed the semaphore would reintroduce exactly
        the connection pile-up the cap exists to prevent: the bank regime
        makes two calls per agent per round, so M=100 would open 200.

        max_tokens=1 and temperature=0 are fixed here rather than passed in.
        The probe is a measurement, not a generation: sampling temperature
        would add noise to the thing being measured, and a second token would
        be read by nothing. run.temperature governs replies only.
        """
        provider = self._provider_name()
        async with self._concurrency_slot():
            await self._rate_limiter.acquire(provider)
            try:
                response = await self._score_with_retry(prompt_text)
            finally:
                self._rate_limiter.release(provider)

        # Always "text": a probe never carries an image. Recorded like any
        # other call, so total_cost_usd counts probes and a cost ceiling can
        # stop a run that is spending on them.
        self._cost_tracker.record(
            provider=provider,
            model_id=_bare_model_id(self.model_backend_id),
            modality="text",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return response

    @retry(
        retry=retry_if_exception_type(TransientGatewayError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _score_with_retry(self, prompt_text: str) -> BackendResponse:
        started = time.monotonic()
        try:
            raw = await litellm.acompletion(
                model=self.model_backend_id,
                messages=_build_messages(prompt_text, None),
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=self._top_logprobs,
                api_base=self._api_base,
            )
        except litellm.exceptions.RateLimitError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.APIConnectionError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.AuthenticationError as e:
            raise FatalGatewayError(str(e)) from e
        except litellm.exceptions.ContextWindowExceededError as e:
            # Before BadRequestError, which it subclasses: see
            # ContextLengthExceeded. Fatal for the RUN, not for this agent.
            raise ContextLengthExceeded(str(e)) from e
        except litellm.exceptions.BadRequestError as e:
            # Includes the case where the server does not support logprobs at
            # all. Fatal rather than transient: retrying cannot make a server
            # grow the feature, and the run should stop rather than burn its
            # retry budget three times per agent per round discovering that.
            raise FatalGatewayError(str(e)) from e

        latency_ms = int((time.monotonic() - started) * 1000)
        top_logprobs = _extract_top_logprobs(raw, self.model_backend_id)

        return BackendResponse(
            text=_first_token_text(raw),
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_provider_response=raw.model_dump(),
            top_logprobs=top_logprobs,
        )

    @retry(
        retry=retry_if_exception_type(TransientGatewayError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_with_retry(
        self, prompt_text: str, image: bytes | None, temperature: float
    ) -> BackendResponse:
        started = time.monotonic()
        try:
            raw = await litellm.acompletion(
                model=self.model_backend_id,
                messages=_build_messages(prompt_text, image),
                temperature=temperature,
                api_base=self._api_base,
            )
        except litellm.exceptions.RateLimitError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.APIConnectionError as e:
            raise TransientGatewayError(str(e)) from e
        except litellm.exceptions.AuthenticationError as e:
            raise FatalGatewayError(str(e)) from e
        except litellm.exceptions.ContextWindowExceededError as e:
            # Before BadRequestError, which it subclasses: see
            # ContextLengthExceeded. Fatal for the RUN, not for this agent.
            raise ContextLengthExceeded(str(e)) from e
        except litellm.exceptions.BadRequestError as e:
            raise FatalGatewayError(str(e)) from e

        # Measured here rather than read off the provider response.
        # full_design_doc.md Sec 3.5's sample reads raw.response_ms, which does
        # not exist on litellm's ModelResponse (neither response_ms nor
        # _response_ms is present). That raises AttributeError, which is
        # neither TransientGatewayError nor FatalGatewayError, so per the
        # orchestrator's error contract it would abort the whole run on the
        # first *successful* API response. int() because Interaction.latency_ms
        # is an int with Field(ge=0) and rejects a fractional float.
        latency_ms = int((time.monotonic() - started) * 1000)

        return BackendResponse(
            text=raw.choices[0].message.content,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_provider_response=raw.model_dump(),
        )

    def _concurrency_slot(self):
        """Acquires a semaphore slot, or a no-op when unbounded."""
        if self._semaphore is not None:
            return self._semaphore
        return _null_slot()

    def _provider_name(self) -> str:
        return self.model_backend_id.split("/")[0] if "/" in self.model_backend_id else "openai"


@contextlib.asynccontextmanager
async def _null_slot():
    """Unbounded: no cap, no waiting."""
    yield


def _bare_model_id(model_backend_id: str) -> str:
    """Strips any provider prefix: 'anthropic/claude-sonnet-4-6' ->
    'claude-sonnet-4-6'. Both the vision table and pricing_table.yaml are
    keyed by bare model ids, so every lookup must normalize through here."""
    return model_backend_id.split("/")[-1]


def _resolve_vision_support(model_backend_id: str) -> bool:
    """Static lookup table, not a runtime probe -- checked once at
    ModelGateway construction. Raises ValueError for an unrecognized
    model_backend_id rather than silently defaulting to False, since a
    silent False could cause a vision-capable model to be treated as
    text-only without anyone noticing."""
    base = _bare_model_id(model_backend_id)
    if base in VISION_CAPABLE:
        return True
    if base in TEXT_ONLY:
        return False
    raise ValueError(f"unrecognized model_backend_id, cannot determine vision support: {model_backend_id}")


def _extract_top_logprobs(raw, model_backend_id: str) -> list[tuple[str, float]]:
    """Pulls [(token, logprob), ...] out of a provider response.

    Raises FatalGatewayError when the response carries no logprobs, rather
    than returning an empty list. An empty list would reach probe_scorer,
    which would raise on it anyway -- but as a per-agent ProbeMassTooLow-ish
    failure repeated for every agent on every round, which reads in the output
    like a model that refuses to answer. The actual cause is a server that was
    never going to return logprobs, which is one wiring error and should
    surface as one loud failure.

    Tolerates both shapes litellm hands back: attribute-style objects
    (ChatCompletionTokenLogprob) and the plain dicts some providers produce
    once litellm has round-tripped them through JSON.
    """

    def _fail(detail: str) -> FatalGatewayError:
        return FatalGatewayError(
            f"{model_backend_id} returned no next-token logprobs ({detail}). "
            "The probe cannot be scored without them: confirm the server "
            "supports logprobs/top_logprobs (vLLM does; some proxies strip them)."
        )

    choices = getattr(raw, "choices", None)
    if not choices:
        raise _fail("no choices in the response")

    logprobs = _get(choices[0], "logprobs")
    if logprobs is None:
        raise _fail("choices[0].logprobs is absent")

    content = _get(logprobs, "content")
    if not content:
        raise _fail("choices[0].logprobs.content is empty")

    entries = _get(content[0], "top_logprobs")
    if not entries:
        raise _fail("choices[0].logprobs.content[0].top_logprobs is empty")

    pairs: list[tuple[str, float]] = []
    for entry in entries:
        token = _get(entry, "token")
        logprob = _get(entry, "logprob")
        if token is None or logprob is None:
            raise _fail(f"a top_logprobs entry is missing token or logprob: {entry!r}")
        pairs.append((str(token), float(logprob)))
    return pairs


def _first_token_text(raw) -> str:
    """The sampled token itself, for the log.

    Not used for scoring -- the distribution is what gets scored, and the
    sampled token is only its argmax -- but kept so a probe row shows what the
    model actually emitted, which is how an off-scale answer gets diagnosed.
    """
    content = _get(raw.choices[0], "message")
    text = _get(content, "content") if content is not None else None
    return "" if text is None else str(text)


def _get(obj, name: str):
    """Attribute or key, whichever the provider gave us."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _build_messages(prompt_text: str, image: bytes | None) -> list[dict]:
    if image is None:
        return [{"role": "user", "content": prompt_text}]
    b64 = base64.b64encode(image).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]
