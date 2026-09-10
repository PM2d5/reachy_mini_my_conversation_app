"""Resolve active profile prompts and voice settings."""

import random
import logging
from typing import TYPE_CHECKING
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


if TYPE_CHECKING:
    from my_conversation_app.face_recognition import SessionIdentity


logger = logging.getLogger(__name__)

# Each app start opens a fresh session with no memory of previous greetings, so the
# app varies the style itself; asking the model to "vary" cannot work across sessions.
# Each entry is a style, not a literal line, so every greeting stays natural.
DEFAULT_GREETING_PROMPTS = (
    (
        "Start the conversation now with a brief, spontaneous greeting in character — "
        "warm and clearly glad to see the user, asking how they are doing. Keep it to "
        "one sentence and invite the user in naturally, in the language you speak."
    ),
    (
        "Start the conversation now with a brief, spontaneous greeting in character that "
        "remarks on the moment — the time of day, the occasion, or simply that the user "
        "has arrived. Keep it to one sentence and invite the user in naturally, in the "
        "language you speak."
    ),
    (
        "Start the conversation now with a brief, spontaneous greeting in character — "
        "curious and full of energy, inviting the user to say what they feel like doing "
        "or talking about. Keep it to one sentence, in the language you speak."
    ),
    (
        "Start the conversation now with a brief, spontaneous greeting drawn from your "
        "character's own world — one playful line only you would say, that pulls the "
        "user in. Keep it to one sentence, in the language you speak."
    ),
    (
        "Start the conversation now with a brief, spontaneous greeting in character — "
        "light and humorous, one witty or self-aware line that makes the user smile. "
        "Keep it to one sentence and invite the user in naturally, in the language you speak."
    ),
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

# Same rotation mechanics as the acks above, but personalized: face recognition
# resolved who woke the robot, and the {name} slot lets each flavor greet them
# as a familiar person instead of a stranger. Every wake opens a memoryless
# session, so near-identical "greet them by name" styles all converge to the
# same words — each flavor below must be a distinct register, like the guest
# pool above (observed: four samey styles, one identical 「凯磊，早啊」every time).
# Realtime models have no clock: the {clock} slot carries the actual local time,
# without which any "time of day" greeting is a guess (observed: 22点说早上好).
KNOWN_USER_WAKE_ACKNOWLEDGEMENT_PROMPTS = (
    (
        "The user just woke you with the wake word and you recognize them — their name "
        "is {name}. Answer with a simple spoken call-back: just their name plus one short "
        "acknowledging word, the way you answer a friend calling you — in the language "
        "you speak. No greeting formula."
    ),
    (
        "The user just woke you with the wake word and you recognize them: {name}. It is "
        "now {clock} local time. Answer with one very short spoken greeting that matches "
        "THAT time of day — morning, afternoon, evening, or night, never another — and "
        "includes their name — two or three words at most, in the language you speak. No "
        "full sentence."
    ),
    (
        "The user just woke you with the wake word and you recognize them: {name}. Answer "
        "with one flat, low-energy spoken acknowledgement — their name plus a lazy little "
        "syllable, as if you were dozing comfortably and can't be bothered — two or three "
        "words at most, in the language you speak. No full sentence, no enthusiasm."
    ),
    (
        "The user just woke you with the wake word and you recognize them: {name}. Answer "
        'with one very short playful spoken tease using their name — a light "back again?" '
        'or "what are we doing today?" vibe — two or three words at most, in the language '
        "you speak. No full sentence."
    ),
    (
        "The user just woke you with the wake word and you recognize them: {name}. Answer "
        "with one very short casual spoken half-greeting — a hummed syllable or interjection "
        "followed by their name, as if you were mid-thought — in the language you speak. "
        "No full sentence."
    ),
)


def format_identity_for_prompt(identity: "SessionIdentity | None") -> str:
    """Return the session-identity fragment, empty for unrecognized users."""
    if identity is None:
        return ""
    return (
        f"The person you are talking to is {identity.name} (confirmed by face recognition). "
        "Address them by name naturally — now and then, not every sentence — and treat what "
        "you remember about them as being about them."
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

# Omni realtime models are vision-capable but receive no live video in this app;
# without a hard rule they either hallucinate what they "see" or refuse the
# question instead of calling the camera tool (both observed with
# qwen3.5-omni-flash-realtime, roughly a coin flip under 18-tool competition
# until the few-shot example below was added).
CAMERA_TOOL_RULE = (
    "## VISION RULE (CRITICAL)\n"
    "You DO have a working camera — the `camera` tool IS your eyes. It is the ONLY way "
    "you can see anything. The moment the user asks anything visual — what do you see, "
    "what am I wearing or holding, how do I look, how many people are here, 看看/看到/"
    "你看得见吗 — your FIRST action is to call `camera`. Do not answer from imagination, "
    "do not say you cannot see or have no camera, do not role-play looking, do not ask "
    "the user to describe it for you. Only after the photo returns may you answer, and "
    "only from what the photo shows. If the tool result reports an error, tell the user "
    "the camera is not working right now.\n"
    "When the user asks about a direction — 左边/右边/上面/look left/what's on your "
    "right — first call move_head(direction=...) to turn there, then call camera to "
    "capture that view; never describe a direction from a front-facing photo.\n"
    'Example: the user asks "你看我穿的是什么颜色的衣服" — you call '
    'camera(question="the color of the user\'s clothes") FIRST, then answer from the '
    "photo. Answering that question without calling `camera` first is always wrong."
)


# Same under-selection problem as the vision rule: asked to "make a sad face"
# (弄一个伤心的表情), realtime models voice-act the emotion and never touch
# play_emotion — zero calls across every logged session while move_head,
# camera, and dance all fire. Explicit commands now have a local trigger;
# this rule additionally pushes the model to emote on its own emotional turns.
EMOTION_TOOL_RULE = (
    "## EXPRESSION RULE (CRITICAL)\n"
    "You DO have a face — the `play_emotion` tool IS how you show expressions. "
    "The moment the user asks you to show, make, or perform an emotion, mood, or "
    "gesture — 开心/伤心/难过/生气/害怕/惊讶/无聊/困, 点头/摇头, 做个表情/show me "
    "happy — your FIRST action is to call `play_emotion` with the matching intent "
    "(开心→happy, 伤心/难过→sad, 生气/愤怒→angry, 害怕→scared, 惊讶→surprised, "
    "无聊→bored, 困→sleepy, 点头→yes, 摇头→no; no clear match → random). Only then "
    "reply, briefly, in that emotion's tone.\n"
    "Beyond explicit requests: whenever your reply itself carries a clear, strong "
    "emotion — comforting someone sad, celebrating good news, apologizing, sharing "
    "the user's excitement — call `play_emotion` in that same turn, while you "
    "speak. Skip it on neutral, factual, or task-like turns: a move every turn "
    "feels mechanical, so emote only when the feeling is genuinely the point.\n"
    "Never act an emotion out with your voice alone — the user watches your "
    "head, and answering an expression request without calling `play_emotion` "
    "first is always wrong.\n"
    'Example: the user asks "弄一个伤心的表情" — you call '
    'play_emotion(emotion="sad") FIRST, then say one short sad-toned sentence.\n'
    'Example: the user shares "我的狗狗生病了，我很难过" — you call '
    'play_emotion(emotion="sad") while you speak, then comfort them gently.'
)


def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


def get_session_instructions(
    instance_path: str | Path | None = None,
    identity: "SessionIdentity | None" = None,
) -> str:
    """Return instructions for the active profile with memory and identity context."""
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
    identity_prompt = format_identity_for_prompt(identity)
    # The vision rule leads the instructions: measured 5/5 camera-tool calls for
    # visual questions with qwen3.5-omni-flash-realtime, vs 3/5 at the tail.
    # The expression rule rides right behind it for the same reason.
    parts = [
        part for part in (CAMERA_TOOL_RULE, EMOTION_TOOL_RULE, memory_prompt, identity_prompt, instructions) if part
    ]
    combined = "\n\n".join(parts)
    logger.info(
        "Session instructions: %d chars, vision rule at offset %d", len(combined), combined.find("## VISION RULE")
    )
    return combined


def get_session_voice(default: str | None = None) -> str:
    """Return the active profile voice or the backend default."""
    fallback = get_default_voice() if default is None else default
    try:
        return _active_profile().voice or fallback
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile voice: %s", exc)
        return fallback


def get_session_greeting_prompt() -> str:
    """Return the active profile greeting prompt or a random default flavor."""
    try:
        greeting = _active_profile().greeting
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile greeting: %s", exc)
        greeting = None
    # Random, not a rotating index: the greeting fires once per launch, so an
    # in-memory counter would pin every fresh start to the same flavor.
    return greeting or random.choice(DEFAULT_GREETING_PROMPTS)
