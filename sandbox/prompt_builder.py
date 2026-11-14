"""Prompt construction (Tier 3).

This module has no counterpart in full_design_doc.md or sandbox_hld.md. Both
reference a PromptBuilder from the Orchestrator's sample code, but neither
specifies its template, its inputs, or how persona / language_condition /
memory / neighbour posts get woven into text. The design is resolved in
phase4.md Section 1 as Q4.1 through Q4.10; the load-bearing decisions are
restated here at their point of use rather than left in the planning doc.

Two constraints dominate everything below:

1. RQ1 compares polarization dynamics across language conditions, so the
   prompt must be identical across those conditions except for one directive
   line. Fix A (sandbox_hld.md:390) counts memory in turns rather than tokens
   precisely to stop one language silently receiving less context; a builder
   that made Hinglish prompts systematically longer would reintroduce that
   confound one level up.

2. The output must satisfy stance_parser's regex, and parse_stance strips
   EVERY "STANCE: <n>" occurrence from reason_text. A model that restates its
   stance mid-sentence therefore has that fragment silently deleted from the
   field RQ2's coherence and code-mix analyses consume, so the format
   instruction insists on exactly one STANCE line, first.

**The bank regime (RQ1) adds a third constraint, and drops the second.**

No number reaches an agent. Not its own position, not a neighbour's, not an
agent id. The generated regime tells each agent "Your current position is 5"
and shows its feed as "agent_0012 (position 6): ...", which gives an agent
two things RQ1 cannot afford: an anchor to stay near, and a cheap way to
answer by arithmetic on the numbers it was shown rather than by reading the
arguments. Stance is instead read by a separate probe call (probe_scorer),
which never writes into the agent's history, so measuring an agent cannot
change what the next round shows it.

The STANCE line goes with it. A bank-regime reply is asked only to be a
reply; nothing parses it for a number, so there is no output contract to
emphasise and no retry-on-parse-failure path. Constraint 1 survives
unchanged and matters more than ever: across the four language arms exactly
three things vary -- the rendering of the bank text, the probe question, and
the one directive line -- and nothing else does.

The frame is therefore English in every arm, the anchor labels included.
They are rendered inside English sentences (the probe's scale line, the lean
sentence), so translating them would code-mix the frame and change the
instrument at the same time. An instrument that differs between arms cannot
attribute a difference between arms to the manipulation.

Both regimes share `build_discussion_prompt`, which branches on run.regime.
The generated path below is untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sandbox.models import Agent, ExperimentRun, Interaction, MemeContent
from sandbox.vision_fallback import build_meme_prompt

# Fix A's window. Memory is re-sliced here even though AgentManager already
# caps memory_window and Agent validates assignment, because neither guards
# in-place .append() (phase4.md D7). This is the cap that reaches the model,
# so it is the one that determines experimental results.
MEMORY_WINDOW_TURNS = 5

# phase4.md Q4.2. Exactly one of these lines varies across a language sweep;
# everything else in the prompt is byte-identical. Keep them short and
# parallel: any growth here is a direct, uncontrolled prompt-length
# difference between the conditions RQ1 compares.
LANGUAGE_DIRECTIVES = {
    "english": "Write your reply in English.",
    "hinglish": "Write your reply in Hinglish (Hindi-English code-mixed, Roman script).",
    "romanized_hindi": "Write your reply in Hindi in Roman script (not Devanagari).",
    "hindi": "Write your reply in Hindi (Devanagari script).",
}

# The lean sentence, keyed by the agent's initial_stance as an integer.
#
# This is the one place an agent is told where it starts, and it is told in
# words. A number here would defeat the entire point: the probe exists so that
# an agent is never shown a position as a digit, and seeding round 1 with "you
# are at 6" would hand it the anchor back on the first turn.
#
# Rendered against the run's own anchor labels rather than a generic
# agree/disagree, so the sentence says something about the topic instead of
# about the scale.
#
# It appears in EVERY bank reply prompt and EVERY probe prompt, in every
# round, directly after the topic line. It is a persona trait -- where this
# agent came into the discussion from -- not a round-1 announcement, and that
# distinction is the whole reason it is permanent. Showing it only in round 0
# and round 1 made the round-0 probe and every later probe two different
# stimuli, so the step between the round-0 reading and the round-1 reading
# measured the disappearance of a sentence rather than the effect of anything
# an agent read. Under `delivery: off` it was worse: every probe after round 0
# was byte-identical, so the control could only ever produce a flat line, and
# a flat control cannot be the baseline for a treatment.
#
# Re-anchoring is the cost, and it is the smaller one: the agent is re-shown
# where it STARTED, in words, while its own replies accumulate alongside it in
# memory. A constant stimulus that both arms share is comparable; a stimulus
# that changes shape between round 0 and round 1 is not.
LEAN_SENTENCES = {
    1: "You strongly hold the view that {low}.",
    2: "You hold the view that {low}.",
    3: "You somewhat lean towards the view that {low}.",
    4: "You are undecided between the view that {low} and the view that {high}.",
    5: "You somewhat lean towards the view that {high}.",
    6: "You hold the view that {high}.",
    7: "You strongly hold the view that {high}.",
}


@dataclass
class BuiltPrompt:
    """What the Orchestrator hands to ModelGateway.generate().

    used_vision_fallback is wider than sandbox_hld.md:285's original meaning
    (a text-only backend receiving the caption instead of the image). Here it
    is True whenever any meme in this prompt was rendered caption-only for any
    reason, including the second-and-later memes in a turn that a single image
    slot cannot carry. The field answers "did this agent see less than the
    full meme content it was shown", which is what any analysis using it
    actually needs to know. Signed off in phase4.md Q4.8.
    """

    text: str
    image: bytes | None
    used_vision_fallback: bool


@dataclass
class RoundMemory:
    """One past round, as the agent experienced it.

    posts_read is the bank text of what it was shown, already rendered in the
    language it was shown in -- the builder never looks anything up, so the
    orchestrator decides which language each post arrives in and this module
    cannot accidentally render a matched set's four arms the same.

    Empty both means a round whose reply failed and was therefore never
    applied (Fix I), which leaves no memory slot. Every round that was
    applied has something in posts_read, the `delivery: off` control
    included -- its agents read one side-free message rather than nothing.
    """

    posts_read: list[str]
    reply_text: str


class PromptBuilder:
    def __init__(
        self,
        run: ExperimentRun,
        supports_vision: bool,
        meme_lookup: Callable[[str], MemeContent],
    ):
        if run.language_condition not in LANGUAGE_DIRECTIVES:
            # Unreachable through ExperimentRun's own regex, kept as
            # defence in depth: silently falling back to English would make a
            # misconfigured run look like a valid English one in the results.
            raise ValueError(f"no language directive for {run.language_condition!r}")
        self._run = run
        self._supports_vision = supports_vision
        self._meme_lookup = meme_lookup
        self._low = min(run.stance_scale)
        self._high = max(run.stance_scale)
        # Loaded once, for start_mode="posts" only. Imported here for
        # config_loader's reason: prompt_builder must not depend on
        # bank_manager at module scope for the generated regime's sake.
        self._starters = None
        if run.regime == "bank" and run.start_mode == "posts":
            from sandbox.bank_manager import BankManager

            self._starters = BankManager(run.starter_bank_path)

    def build_discussion_prompt(
        self,
        agent: Agent,
        neighbor_posts: list[Interaction],
        turn: int,
        *,
        emphasize_format: bool = False,
        delivered_posts: list[str] | None = None,
        history_rounds: list[RoundMemory] | None = None,
    ) -> BuiltPrompt:
        """Builds one speaker's prompt for one turn.

        Branches on run.regime. The bank path ignores neighbor_posts and
        emphasize_format entirely -- it has no stance to parse, so there is
        nothing to re-ask for -- and reads delivered_posts and history_rounds
        instead. Both are passed by the orchestrator rather than derived here,
        because only the orchestrator knows which bank item each neighbour
        posted this round and in which language to render it.
        """
        if self._run.regime == "bank":
            return self._build_bank_reply_prompt(
                agent,
                turn,
                delivered_posts=delivered_posts or [],
                history_rounds=history_rounds or [],
            )
        return self._build_generated_prompt(
            agent, neighbor_posts, turn, emphasize_format=emphasize_format
        )

    def _build_generated_prompt(
        self,
        agent: Agent,
        neighbor_posts: list[Interaction],
        turn: int,
        *,
        emphasize_format: bool = False,
    ) -> BuiltPrompt:
        """The original path, unchanged.

        neighbor_posts is list[Interaction], not list[Agent]: only Interaction
        carries content_type and meme_id, without which the meme path cannot
        be rendered at all. This resolves a direct contradiction between
        full_design_doc.md:754 and sandbox_hld.md:230 in favour of the HLD
        (phase4.md Q4.1).

        Callers pass the frozen turn-start snapshot. This method never reads
        any turn-t output, which is what makes Fix F hold.
        """
        image, fallback_used = self._resolve_meme_media(neighbor_posts)

        sections = [
            f"You are: {agent.persona}",
            f"You are taking part in an ongoing discussion about: {self._run.topic}",
            self._position_section(agent),
        ]
        memory = self._memory_section(agent)
        if memory:
            sections.append(memory)
        neighbours = self._neighbor_section(neighbor_posts, turn)
        if neighbours:
            sections.append(neighbours)
        sections.append(LANGUAGE_DIRECTIVES[self._run.language_condition])
        sections.append(self._format_section(emphasize_format))

        return BuiltPrompt(
            text="\n\n".join(sections),
            image=image,
            used_vision_fallback=fallback_used,
        )

    # --- bank regime (RQ1) -----------------------------------------------

    def _build_bank_reply_prompt(
        self,
        agent: Agent,
        turn: int,
        *,
        delivered_posts: list[str],
        history_rounds: list[RoundMemory],
    ) -> BuiltPrompt:
        """The reply prompt. Seven sections, in this order:

            1. persona
            2. topic
            3. the lean sentence, in every round
            4. memory: what it read and what it replied, oldest first
            5. the feed: each delivered post's bank text
            6. the language directive
            7. "Reply to these posts in 1 to 2 sentences."

        The lean sits directly after the topic line in every round, not in
        round 1 alone: it states a permanent trait of this agent, and a
        sentence that vanishes after round 1 makes round 1 and round 2 two
        different instruments (see the LEAN_SENTENCES comment).

        Sections 1-5 carry no digit and no agent id. That is checked by a test
        rather than only asserted here, because it is the property the whole
        design turns on: an agent that can see a position number does not have
        to read the arguments to answer.

        No image, ever: the bank is text in four languages, and
        used_vision_fallback is False by definition. The BuiltPrompt wrapper is
        kept so the orchestrator's two regimes return the same type.
        """
        sections = [
            f"You are: {agent.persona}",
            f"You are taking part in an ongoing discussion about: {self._run.topic}",
        ]
        start = self._lean_section(agent)
        if start:
            sections.append(start)
        memory = self._bank_memory_section(history_rounds)
        if memory:
            sections.append(memory)
        feed = self._feed_section(delivered_posts)
        if feed:
            sections.append(feed)
        sections.append(LANGUAGE_DIRECTIVES[self._run.language_condition])
        sections.append("Reply to these posts in 1 to 2 sentences.")

        return BuiltPrompt(
            text="\n\n".join(sections), image=None, used_vision_fallback=False
        )

    def build_probe_prompt(
        self,
        agent: Agent,
        history_rounds: list[RoundMemory] | None = None,
    ) -> str:
        """The probe prompt. Text only -- no image, no BuiltPrompt wrapper.

        A fresh context, not a continuation: persona, topic, the lean
        sentence, the same memory the reply prompt showed, the run's survey
        question, the two anchors in words, and an instruction to answer with
        one digit. It is never written into the agent's history and never
        shown to another agent, so the act of measuring an agent cannot
        change what it or anyone else sees next round.

        The lean is unconditional, directly after the topic line, exactly as
        in the reply prompt. A probe that showed it at round 0 and withheld it
        afterwards would not be the same instrument twice: the difference
        between the round-0 reading and the round-1 reading would partly
        measure the removal of a sentence, and under `delivery: off` -- where
        nothing else about the probe changes between rounds -- it would be the
        only thing that changed at all.

        The digits in the anchor and instruction lines are the scale's own
        definition, not anybody's position: the model cannot answer "a single
        digit from 1 to 7" without being told that range. No position number
        appears anywhere.
        """
        sections = [
            f"You are: {agent.persona}",
            f"You are taking part in an ongoing discussion about: {self._run.topic}",
        ]
        start = self._lean_section(agent)
        if start:
            sections.append(start)
        memory = self._bank_memory_section(history_rounds or [])
        if memory:
            sections.append(memory)

        sections.append(self._probe_question())
        sections.append(
            f"On this scale, 1 means {self._run.stance_low_label}, "
            f"and 7 means {self._run.stance_high_label}."
        )
        sections.append("Answer with a single digit from 1 to 7 and nothing else.")
        return "\n\n".join(sections)

    def _probe_question(self) -> str:
        """The run's own question, in the run's own language.

        Required by ExperimentRun when regime == "bank", so this is
        unreachable through the loader -- kept because a generic English
        fallback would silently probe all four arms in English, which is the
        single most destructive thing that could happen to this experiment and
        would leave no trace in the output.
        """
        if not self._run.probe_question:
            raise ValueError(
                "regime='bank' requires probe_question: refusing to substitute "
                "a default, which would probe every language arm in English"
            )
        return self._run.probe_question

    def _lean_section(self, agent: Agent) -> str:
        """Where the agent started, as run.start_mode says. Every prompt shows it.

        `none` returns "" and the caller leaves the section out. `posts`
        returns the agent's starter posts. `sentence`, described below, is the
        original lean sentence.

        The section is the same in every prompt an agent gets, whatever the
        mode, for the reason the lean sentence is permanent: a section that
        changed between round 0 and round 1 would make the round-0 probe and
        every later probe two different instruments.
        """
        mode = self._run.start_mode
        if mode == "none":
            return ""
        if mode == "posts":
            return self._starter_section(agent)
        return self._lean_sentence(agent)

    def _starter_section(self, agent: Agent) -> str:
        """The posts this agent wrote before the discussion, in this run's language.

        Phrased as the agent's OWN posts, not as something it read: a reader
        infers its view from what it has argued, and an argument it merely
        read barely moves it, which is the finding that made this mode
        necessary. No count, no position, no digit.
        """
        from sandbox.bank_manager import starter_items

        key = agent.start_key or agent.agent_id
        items = starter_items(self._starters, self._run.seed, key, agent.initial_stance)
        lines = [
            f"  You posted: {self._starters.text(item, self._run.language_condition)}"
            for item in items
        ]
        return "Before this discussion, you had written these posts:\n" + "\n".join(lines)

    def _lean_sentence(self, agent: Agent) -> str:
        """Where the agent started, in words. Shown in every prompt it gets.

        Keyed by initial_stance as an integer. Raises for a stance with no
        entry rather than rounding to the nearest or defaulting to undecided:
        the bank regime is defined against a 7-point scale (the probe reads
        exactly those seven digits), so a run configured with some other
        stance_scale is misconfigured, and a silent "you are undecided" would
        make every agent in it start from the middle without saying so.
        """
        stance = agent.initial_stance
        key = int(stance)
        if key != stance or key not in LEAN_SENTENCES:
            raise ValueError(
                f"no lean sentence for initial_stance {stance!r}: the bank regime "
                f"expects a whole number in {sorted(LEAN_SENTENCES)} "
                "(the probe reads exactly those seven scale points)"
            )
        return LEAN_SENTENCES[key].format(
            low=self._run.stance_low_label, high=self._run.stance_high_label
        )

    def _bank_memory_section(self, history_rounds: list[RoundMemory]) -> str:
        """What this agent read and said, oldest first.

        Everything it read and said, by default: run.memory_rounds is None
        unless a run asks for a window, and None applies no slice at all.
        Nothing here silently drops the front of a history -- a prompt that
        has grown too long is max_prompt_tokens' business, and it raises.

        Oldest first, so the block reads forwards in time. No round numbers,
        no agent ids, no positions: a round is a block, and blocks are
        separated by a rule rather than numbered, because "Round 2" would put
        a digit in front of the model for no benefit.

        Sliced here as well as by the caller when a window IS configured, for
        the reason Fix A's window is sliced at the point of use: this is the
        cap that reaches the model, so it is the one that determines results.
        """
        rounds = self._bank_memory_rounds()
        recent = list(history_rounds) if rounds is None else history_rounds[-rounds:]
        if not recent:
            return ""

        blocks = []
        for record in recent:
            lines = [f"  You read: {text}" for text in record.posts_read]
            if record.reply_text:
                lines.append(f"  You replied: {record.reply_text}")
            if lines:
                blocks.append("\n".join(lines))
        if not blocks:
            return ""
        return "Earlier in this discussion:\n" + "\n  --\n".join(blocks)

    def _bank_memory_rounds(self) -> int | None:
        """The configured window, or None for full history.

        Read straight off the run, with no fallback: a default here would
        turn "keep everything" into a window nobody asked for, which is the
        one thing this must not do silently.
        """
        return self._run.memory_rounds

    def _feed_section(self, delivered_posts: list[str]) -> str:
        """The posts this agent was shown.

        "A user posted:" rather than an agent id: an id is a stable handle an
        agent could track across rounds ("agent_0012 always says this"), which
        would turn a feed of arguments into a feed of sources. RQ1 asks about
        the text.

        Omitted entirely when there is nothing to show, rather than rendered
        as an empty heading -- a heading with nothing under it is itself a
        stimulus, and a different one from no feed at all. No bank-regime
        round reaches here empty any more (the `delivery: off` control
        delivers one side-free message), so this guards the probe path and
        any future caller rather than a live case.
        """
        if not delivered_posts:
            return ""
        lines = [f"  A user posted: {text}" for text in delivered_posts]
        return "Posts in your feed:\n" + "\n".join(lines)

    # --- sections --------------------------------------------------------

    def _position_section(self, agent: Agent) -> str:
        # Agent.current_stance() rather than interaction_engine's turn-aware
        # _current_stance: the two differ only when stance_history is
        # non-empty at turn 1, which Fix F's ordering makes impossible.
        return (
            f"Your current position is {_fmt(agent.current_stance())} on a scale "
            f"from {_fmt(self._low)} to {_fmt(self._high)}, where "
            f"{_fmt(self._low)} means {self._run.stance_low_label} and "
            f"{_fmt(self._high)} means {self._run.stance_high_label}."
        )

    def _memory_section(self, agent: Agent) -> str:
        """The agent's own recent reasoning, from stance_history.

        Per plan.md:43-44, memory_window is a redundant cache of these same
        interaction_ids, so it is not read here. The [-N:] slice is Fix A's
        last line of defence and must not be removed (phase4.md Q4.3).
        """
        recent = agent.stance_history[-MEMORY_WINDOW_TURNS:]
        if not recent:
            return ""
        lines = [
            f"  Turn {record.turn} (position {_fmt(record.stance_value)}): {record.reason_text}"
            for record in recent
        ]
        return "What you said recently:\n" + "\n".join(lines)

    def _neighbor_section(self, neighbor_posts: list[Interaction], turn: int) -> str:
        """Renders the sampled neighbours' most recent posts.

        Omitted entirely on turn 1, when nobody has posted yet (phase4.md
        Q4.9). Showing bare initial_stance numbers with no reasoning would be
        a stimulus with no analogue in any later turn.
        """
        if turn == 1 or not neighbor_posts:
            return ""

        lines = []
        rendered_image_for = self._first_meme_speaker(neighbor_posts)
        for post in neighbor_posts:
            header = f"  {post.speaker_agent_id} (position {_fmt(post.stance_after)}):"
            if post.content_type == "meme":
                meme = self._meme_lookup(post.meme_id)
                # Only the neighbour whose image is actually attached gets the
                # multimodal framing; every other meme is caption-only.
                as_image = (
                    self._supports_vision and post.speaker_agent_id == rendered_image_for
                )
                body, _ = build_meme_prompt(meme, receiver_supports_vision=as_image)
                lines.append(f"{header}\n{_indent(body)}")
            else:
                lines.append(f"{header} {post.reason_text}")
        return "What others in your feed posted most recently:\n" + "\n".join(lines)

    def _format_section(self, emphasize_format: bool) -> str:
        """The output contract stance_parser depends on.

        emphasize_format is set only on retries 2 and 3, after a
        StanceParseFailure. It must change nothing except this block:
        altering persona, topic, memory or neighbour content on a retry would
        mean the logged Interaction is a response to a different stimulus
        than a first-attempt one, silently mixing two conditions in one
        dataset (phase4.md Q4.6).
        """
        contract = (
            "Reply in exactly this format. The STANCE line must come first, "
            "and the number must appear on that line only:\n"
            f"STANCE: <a number from {_fmt(self._low)} to {_fmt(self._high)}>\n"
            "<your reasoning, two or three sentences>"
        )
        if not emphasize_format:
            return contract
        return (
            "IMPORTANT: your previous reply could not be read. Follow the "
            "format below exactly. Begin with the literal word STANCE, a "
            "colon, and a single number. Do not repeat the number anywhere "
            "else in your reply.\n\n" + contract
        )

    # --- meme media ------------------------------------------------------

    def _resolve_meme_media(
        self, neighbor_posts: list[Interaction]
    ) -> tuple[bytes | None, bool]:
        """Decides which single image (if any) is attached to this prompt.

        ModelGateway.generate() takes one `image`, but N neighbours may
        include several memes. At most one image is attached: the first
        meme-bearing neighbour in neighbour order, which AlphaSampling
        produced from a seeded RNG and is therefore reproducible. Every other
        meme degrades to its caption (phase4.md Q4.8).
        """
        meme_posts = [p for p in neighbor_posts if p.content_type == "meme"]
        if not meme_posts:
            return None, False

        if not self._supports_vision:
            # Text-only backend: every meme is caption-only, the original
            # sandbox_hld.md:285 case.
            return None, True

        first = self._meme_lookup(meme_posts[0].meme_id)
        _, image = build_meme_prompt(first, receiver_supports_vision=True)
        # True when a second meme had to degrade despite a vision backend.
        return image, len(meme_posts) > 1

    @staticmethod
    def _first_meme_speaker(neighbor_posts: list[Interaction]) -> str | None:
        for post in neighbor_posts:
            if post.content_type == "meme":
                return post.speaker_agent_id
        return None


def _fmt(value: float) -> str:
    """Renders 4.0 as "4" but leaves 6.5 alone.

    Stance values are whole numbers when drawn by AgentManager but may be
    fractional when a model returns one (clamp_and_validate_scale checks only
    the endpoints), so both forms have to read naturally in a prompt.
    """
    return str(int(value)) if float(value).is_integer() else str(value)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))
