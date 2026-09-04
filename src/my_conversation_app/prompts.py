"""Resolve active profile prompts and voice settings."""

import logging
from pathlib import Path

from my_conversation_app.config import config, get_default_voice
from my_conversation_app.memory import format_memory_for_prompt
from my_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileDefinition,
    ProfileFormatError,
    read_profile,
    read_packaged_default_profile,
)


logger = logging.getLogger(__name__)

DEFAULT_GREETING_PROMPT = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)

# Each wake opens a fresh session with no memory of previous acknowledgements, so the
# app rotates these flavors itself; asking the model to "vary" cannot work across sessions.
WAKE_ACKNOWLEDGEMENT_PROMPTS = (
    (
        "The user just woke you with the wake word. Answer with one very short spoken "
        "acknowledgement — two or three words at most, the way someone answers when their "
        "name is called — in the language you speak. No full sentence."
    ),
    (
        "The user just woke you with the wake word. Answer with one very short spoken "
        "check-in — two or three words at most, casually asking what they need — in the "
        "language you speak. No full sentence."
    ),
    (
        "The user just woke you with the wake word. Answer with a single short spoken "
        "interjection — one or two words, a curious hum or half-word — in the language "
        "you speak. No full sentence."
    ),
    (
        "The user just woke you with the wake word. Answer with one blunt, slightly "
        "impatient spoken 'what?' — a single word, as if they had interrupted you "
        "mid-thought — in the language you speak. No full sentence."
    ),
)

# Same memoryless-session problem as wake acks: the app rotates these wait-line
# styles itself. Each entry is a style, not a literal line, so every voicing
# stays natural while never repeating the previous flavor.
ASSISTANT_WAIT_ACKNOWLEDGEMENT_PROMPTS = (
    (
        "You just started asking your home assistant for help and the user must wait. "
        "Say one short spoken line — under ten words — telling the user to hang on "
        "while you check, in the language you speak. Casual, no explanations."
    ),
    (
        "You just started asking your home assistant for help and the user must wait. "
        "Say one short playful spoken line — under ten words — as if handing the "
        "question to a coworker, in the language you speak. No explanations."
    ),
    (
        "You just started asking your home assistant for help and the user must wait. "
        "Say one short brisk spoken line — under ten words — brisk and efficient, "
        "like a professional taking a task, in the language you speak. No explanations."
    ),
    (
        "You just started asking your home assistant for help and the user must wait. "
        "Say one short self-aware spoken line — under ten words — with light robot "
        "humor about needing a moment, in the language you speak. No explanations."
    ),
    (
        "You just started asking your home assistant for help and the user must wait. "
        "Say one short warm spoken line — under ten words — reassuring the user the "
        "answer is coming, in the language you speak. No explanations."
    ),
)

# The wait line above is injected as a user message and stays in the context; without
# this counter-anchor the model keeps answering it instead of relaying the tool result.
ASSISTANT_RESULT_RELAY_PROMPT = (
    "The home-assistant query just finished and its output is the tool result above. "
    "The wait is over. Relay that answer to the user now, faithfully and concisely, in "
    "the user's language — if the tool result reports an error instead, tell the user "
    "it didn't work out this time. Never announce waiting or checking again."
)


def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Return instructions for the active profile with memory context."""
    selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
    profile_name = selected_profile or DEFAULT_PROFILE_NAME
    try:
        profile = _active_profile()
        instructions = profile.instructions.strip()
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load profile %r: %s", profile_name, exc)
        instructions = ""

    if not instructions and selected_profile and selected_profile != DEFAULT_PROFILE_NAME:
        logger.warning("Using bundled default instructions because profile %r is incomplete", selected_profile)
        try:
            instructions = read_packaged_default_profile().instructions.strip()
        except (FileNotFoundError, ProfileFormatError) as exc:
            raise RuntimeError("Default profile has no usable instructions") from exc
    if not instructions:
        raise RuntimeError("Default profile has no usable instructions")

    memory_prompt = format_memory_for_prompt(instance_path)
    if memory_prompt:
        return f"{memory_prompt}\n\n{instructions}"
    return instructions


def get_session_voice(default: str | None = None) -> str:
    """Return the active profile voice or the backend default."""
    fallback = get_default_voice() if default is None else default
    try:
        return _active_profile().voice or fallback
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile voice: %s", exc)
        return fallback


def get_session_greeting_prompt() -> str:
    """Return the active profile greeting prompt or the app default."""
    try:
        return _active_profile().greeting or DEFAULT_GREETING_PROMPT
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile greeting: %s", exc)
        return DEFAULT_GREETING_PROMPT
