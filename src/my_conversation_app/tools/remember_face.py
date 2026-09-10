"""Enroll the person in front of the camera so Reachy recognizes them later."""

import re
import time
import asyncio
import logging
from typing import Any, Dict

from my_conversation_app.config import config
from my_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# One control-loop tick of the movement manager, plus the poll interval while a
# commanded motion is still settling (same values as the camera tool).
_MOTION_SETTLE_POLL_S = 0.05
_ENROLL_FRAME_COUNT = 3
_ENROLL_FRAME_GAP_S = 0.4
# The local trigger and a model tool call can fire for the same utterance;
# collapse them like PlayEmotion does, so one name never enrolls twice in a row.
_REENROLL_DEDUPE_WINDOW_S = 60.0

# 「记住我喜欢喝咖啡」是记忆请求而非注册：命令后紧跟这些词时必须整体不命中。
_REMEMBER_ME_NEGATIVE_LOOKAHEAD = r"(?!喜欢|爱|这|那|的|一下|之后|以后)"
# 名字与命令之间除标点外还常有连接词：「我叫凯蕾，帮我记住我」。
_COMMAND_GAP = r"[\s,，。!！?？]*(?:请|帮我|帮忙|麻烦你?|快|赶紧|马上|一定要?)*[\s,，。!！?？]*"

_ENROLLMENT_COMMAND_PATTERNS = (
    # 名字在前：「我叫凯蕾，记住我」
    re.compile(
        r"(?:我叫|我是|我的名字是|名字是|叫我)(?P<name>[\w·]{1,16}?)"
        + _COMMAND_GAP
        + rf"(?:记住我|记住我的脸|记住这张脸|记住我这个人){_REMEMBER_ME_NEGATIVE_LOOKAHEAD}(?:吧|啊|哈)?"
    ),
    # 指令在前：「记住我，我叫凯蕾」— 名字须顶到句尾或标点，否则非贪婪会截短
    re.compile(
        rf"(?:记住我|记住我的脸|记住这张脸|记住我这个人){_REMEMBER_ME_NEGATIVE_LOOKAHEAD}(?:吧|啊|哈)?"
        + _COMMAND_GAP
        + r"(?:我叫|我是|我的名字是|名字是|叫我)(?P<name>[\w·]{1,16}?)(?=\s*[,，。!！?？]|\s*$)"
    ),
    # English, name first: "my name is Anna, remember me". "i am|i'm" stay
    # case-sensitive with a capitalized name — "I'm leaving, remember me" is a
    # statement, not an introduction, and predicates stay lowercase.
    re.compile(
        r"(?:my name is|call me)\s+(?P<name>[A-Za-z][\w·]*(?:\s+[A-Za-z][\w·]*){0,2})"
        r"\s*[,.\s]{0,3}\s*(?:please\s+)?(?:remember me|remember my face|remember this face)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:I am|I'm)\s+(?P<name>[A-Z][\w·]*(?:\s+[A-Z][\w·]*){0,2})"
        r"\s*[,.\s]{0,3}\s*(?:please\s+)?(?:remember me|remember my face|remember this face)\b"
    ),
    # English, command first: "remember me, my name is Anna"
    re.compile(
        r"(?:please\s+)?(?:remember me|remember my face|remember this face)\b\s*[,.\s]{0,3}\s*"
        r"(?:my name is|call me)\s+(?P<name>[A-Za-z][\w·]*(?:\s+[A-Za-z][\w·]*){0,2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:please\s+)?(?:remember me|remember my face|remember this face)\b\s*[,.\s]{0,3}\s*"
        r"(?:I am|I'm)\s+(?P<name>[A-Z][\w·]*(?:\s+[A-Z][\w·]*){0,2})\b"
    ),
)

_NOT_A_NAME_WORDS = frozenset({"我", "你", "他", "她", "它"})
# 名字与命令之间没有标点时，连接词和语气词会被非贪婪名字捕获吞进尾部，剥掉它们。
_NAME_FILLER_SUFFIXES = (
    "帮我",
    "帮忙",
    "赶紧",
    "马上",
    "一定",
    "麻烦",
    "请",
    "快",
    "要",
    "得",
    "吧",
    "啊",
    "哈",
    "呀",
    "哦",
    "嘞",
    "了",
)


def _strip_name_fillers(name: str) -> str:
    """Drop connector/particle words glued to the end of a captured name."""
    stripped = True
    while stripped and name:
        stripped = False
        for suffix in _NAME_FILLER_SUFFIXES:
            if name != suffix and name.endswith(suffix):
                name = name[: -len(suffix)]
                stripped = True
                break
    return name


_ENROLLMENT_ERRORS = {
    "too_few_faces": "the camera could not see a clear face; ask the user to look at you and try again",
    "duplicate_name": "this name is already enrolled",
    "store_full": "the face store is full; remove someone first",
    "empty_name": "name must be a non-empty string",
    "no_embeddings": "no usable face images were captured",
}


def match_face_enrollment_command(transcript: str) -> str | None:
    """Return the name when the speaker introduces themselves and asks to be remembered."""
    text = transcript.strip()
    if not text:
        return None
    for pattern in _ENROLLMENT_COMMAND_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        name = " ".join(_strip_name_fillers(match.group("name")).split()).strip()
        if not name or name.lower() in _NOT_A_NAME_WORDS:
            return None
        return name[:24]
    return None


class RememberFace(Tool):
    """Register the person talking to Reachy for face recognition."""

    name = "remember_face"
    description = (
        "Register the FACE of the person talking to you, so later conversations greet them "
        "by name. Call this only when the user explicitly asks you to remember THEM "
        "personally AND gives their name — e.g. 我叫凯蕾，记住我 / remember me, my name is "
        "Anna. If they only say 记住我 without a name, ask for their name first, then call "
        "this. For facts, preferences, or events use the `remember` tool instead — this "
        "tool is only for WHO the person is. Recognition afterwards is automatic; never "
        "call this to identify someone."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "The person's name exactly as they said it, e.g. 凯蕾 or Anna. Keep it under 24 characters."
                ),
            },
        },
        "required": ["name"],
    }

    _last_enrollment: tuple[str, float] | None = None

    def is_available(self) -> bool:
        """Hide the tool when face recognition is switched off."""
        return bool(config.FACE_RECOGNITION_ENABLED)

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Capture a few frames and enroll the dominant face under the given name."""
        name = " ".join(str(kwargs.get("name") or "").split()).strip()
        if not name:
            logger.warning("remember_face: empty name")
            return {"error": _ENROLLMENT_ERRORS["empty_name"]}

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}
        recognizer = deps.face_recognizer
        if recognizer is None or not config.FACE_RECOGNITION_ENABLED:
            return {"error": "Face recognition is not available"}
        if not await asyncio.to_thread(recognizer.load_models):
            return {"error": "Face recognition is not available"}

        now = time.monotonic()
        last = RememberFace._last_enrollment
        if last is not None and last[0].lower() == name.lower() and now - last[1] < _REENROLL_DEDUPE_WINDOW_S:
            logger.info("remember_face: %r enrolled moments ago; skipping duplicate call", name)
            return {"status": "already_enrolled", "name": name}

        logger.info("Tool call: remember_face name=%s", name[:40])

        # Same settle wait as the camera tool, so the head is not still turning.
        await asyncio.sleep(_MOTION_SETTLE_POLL_S)
        while deps.movement_manager.is_moving():
            await asyncio.sleep(_MOTION_SETTLE_POLL_S)

        frames = []
        for _ in range(_ENROLL_FRAME_COUNT):
            frames.append(await asyncio.to_thread(deps.reachy_mini.media.get_frame))
            await asyncio.sleep(_ENROLL_FRAME_GAP_S)
        captured = [frame for frame in frames if frame is not None]

        outcome = await asyncio.to_thread(recognizer.enroll, name, captured)
        if outcome.face is None:
            reason = outcome.reason or "unknown"
            logger.warning("Face enrollment for %r failed: %s", name, reason)
            return {"error": _ENROLLMENT_ERRORS.get(reason, f"enrollment failed ({reason})")}
        RememberFace._last_enrollment = (outcome.face.name, time.monotonic())
        return {"saved": outcome.face.name, "face_id": outcome.face.id}
