import re
import time
import random
import logging
import unicodedata
from typing import TYPE_CHECKING, Any, Dict

from my_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini.motion.recorded_move import RecordedMoves


logger = logging.getLogger(__name__)

try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from my_conversation_app.dance_emotion_moves import EmotionQueueMove

    EMOTION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Emotion library not available: {e}")
    EMOTION_AVAILABLE = False


EMOTION_INTENTS: tuple[str, ...] = (
    "random",
    "happy",
    "excited",
    "loving",
    "grateful",
    "success",
    "thinking",
    "attentive",
    "confused",
    "uncertain",
    "sad",
    "downcast",
    "lonely",
    "angry",
    "irritated",
    "displeased",
    "disgusted",
    "scared",
    "anxious",
    "surprised",
    "amazed",
    "calming",
    "relief",
    "impatient",
    "embarrassed",
    "bored",
    "tired",
    "sleepy",
    "yes",
    "yes_understanding",
    "no",
    "no_sad",
    "no_excited",
    "no_firm",
    "welcoming",
    "greeting",
    "goodbye",
    "go_away",
    "helpful",
    "dance",
    "electric",
    "dying",
)

_EXCELLENT_MOVES: tuple[str, ...] = (
    "anxiety1",
    "boredom2",
    "dance2",
    "dance3",
    "downcast1",
    "dying1",
    "exhausted1",
    "grateful1",
    "helpful1",
    "loving1",
    "rage1",
    "reprimand1",
    "resigned1",
    "sad1",
    "sad2",
    "scared1",
    "sleep1",
    "surprised1",
    "thoughtful1",
    "welcoming2",
)

_OK_CLEAR_MOVES: tuple[str, ...] = (
    "amazed1",
    "attentive1",
    "attentive2",
    "boredom1",
    "confused1",
    "disgusted1",
    "displeased1",
    "displeased2",
    "fear1",
    "impatient2",
    "irritated1",
    "irritated2",
    "laughing1",
    "laughing2",
    "lonely1",
    "no1",
    "no_excited1",
    "no_sad1",
    "reprimand2",
    "shy1",
    "success1",
    "success2",
    "surprised2",
    "thoughtful2",
    "uncertain1",
    "understanding2",
    "yes1",
)

_CURATED_DEFAULT_MOVES: tuple[str, ...] = _EXCELLENT_MOVES + _OK_CLEAR_MOVES

_INTENT_TO_MOVES: dict[str, tuple[str, ...]] = {
    "happy": ("laughing2", "laughing1"),
    "excited": ("dance3", "dance2"),
    "loving": ("loving1",),
    "grateful": ("grateful1",),
    "success": ("success1", "success2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1",),
    "uncertain": ("uncertain1",),
    "sad": ("sad1", "sad2", "downcast1"),
    "downcast": ("downcast1", "sad1"),
    "lonely": ("lonely1",),
    "angry": ("rage1", "irritated2", "irritated1"),
    "irritated": ("irritated1", "irritated2", "displeased2"),
    "displeased": ("displeased1", "displeased2"),
    "disgusted": ("disgusted1",),
    "scared": ("scared1", "fear1", "anxiety1"),
    "anxious": ("anxiety1", "fear1", "scared1"),
    "surprised": ("surprised1", "surprised2", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "calming": ("calming1",),
    "relief": ("relief1", "relief2"),
    "impatient": ("impatient2",),
    "embarrassed": ("shy1",),
    "bored": ("boredom2", "boredom1"),
    "tired": ("exhausted1", "sleep1"),
    "sleepy": ("sleep1", "exhausted1"),
    "yes": ("yes1", "understanding2"),
    "yes_understanding": ("understanding2",),
    "no": ("no1",),
    "no_sad": ("no_sad1",),
    "no_excited": ("no_excited1",),
    "no_firm": ("no1",),
    "welcoming": ("welcoming2",),
    "greeting": ("welcoming2",),
    "goodbye": ("loving1", "welcoming2"),
    "go_away": ("go_away1",),
    "helpful": ("helpful1",),
    "dance": ("dance2", "dance3"),
    "electric": ("electric1",),
    "dying": ("dying1",),
}

_ALLOWED_MOVE_NAMES: frozenset[str] = frozenset(_CURATED_DEFAULT_MOVES).union(*_INTENT_TO_MOVES.values())

_KEYWORD_INTENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("no", "sad"), "no_sad"),
    (("no", "excited"), "no_excited"),
    (("no", "firm"), "no_firm"),
    (("yes", "understanding"), "yes_understanding"),
)

# Local trigger: realtime models (notably on DashScope) voice-act explicit
# expression requests instead of calling the tool, in any language, so the
# app matches the command itself and queues the move deterministically.
_EXPRESSION_COMMAND_INTENTS: tuple[tuple[str, str], ...] = (
    ("开心", "happy"),
    ("高兴", "happy"),
    ("快乐", "happy"),
    ("兴奋", "excited"),
    ("伤心", "sad"),
    ("难过", "sad"),
    ("悲伤", "sad"),
    ("委屈", "sad"),
    ("沮丧", "downcast"),
    ("失落", "downcast"),
    ("低落", "downcast"),
    ("生气", "angry"),
    ("愤怒", "angry"),
    ("恼火", "angry"),
    ("厌恶", "disgusted"),
    ("嫌弃", "disgusted"),
    ("害怕", "scared"),
    ("恐惧", "scared"),
    ("焦虑", "anxious"),
    ("紧张", "anxious"),
    ("惊讶", "surprised"),
    ("吃惊", "surprised"),
    ("震惊", "amazed"),
    ("无聊", "bored"),
    ("瞌睡", "sleepy"),
    ("想睡", "sleepy"),
    ("困", "sleepy"),
    ("疲惫", "tired"),
    ("疲倦", "tired"),
    ("累", "tired"),
    # 不同意 must precede 同意: the shorter word would otherwise match inside it.
    ("不同意", "no"),
    ("同意", "yes"),
    ("害羞", "embarrassed"),
    ("尴尬", "embarrassed"),
    ("孤独", "lonely"),
    ("孤单", "lonely"),
    ("喜爱", "loving"),
    ("感谢", "grateful"),
    ("感激", "grateful"),
    ("欢迎", "welcoming"),
    ("问候", "welcoming"),
    ("再见", "goodbye"),
    ("告别", "goodbye"),
    ("安慰", "calming"),
    ("安抚", "calming"),
    ("放心", "relief"),
    ("不耐烦", "impatient"),
    ("伤感", "sad"),
    ("搞笑", "happy"),
    ("好笑", "happy"),
    ("恐怖", "scared"),
    ("吓人", "scared"),
)

# Words that negate the emotion right before them (不开心 must not hit happy).
# 别 is excluded: it negates verbs (别难过) but ends adverbs (特别惊讶), and a
# comforting "别难过" emoting concern is acceptable anyway.
_NEGATION_PREFIXES = "不没无非"

_EXPRESSION_COMMAND_RE = re.compile(
    r"(?:做|来|弄|表演|展示|秀)一?[个点下段场支]?(?P<emotion_cn>[^，。！？、\s]{0,4}?)的?(?:表情|情绪|动作|emo)"
    r"|(?:do|make|show|give|perform)\s+(?:me\s+)?an?\s+(?P<emotion_en>[\w-]+)?\s+(?:emotion|face|expression)"
    r"|(?:讲|说|来|唱|念|表演)一?[个首段场支]?(?P<performance_emo>[^，。！？、\s]{0,4}?)的?"
    r"(?P<performance_noun>故事|笑话|段子|童话|绕口令|歌曲|曲子|诗|歌)",
    re.IGNORECASE,
)

# Performance nouns that carry an inherent mood when the request names none.
_INHERENT_NOUN_INTENTS: dict[str, str] = {"笑话": "happy", "段子": "happy"}


def match_expression_command(transcript: str) -> str | None:
    """Return the intent for an explicit show-an-expression or perform-a-mood command, else None.

    Deliberately narrow: only imperative command forms (做个伤心的表情 / 讲一个
    悲伤的故事 / do a sad face) match, never emotional small talk. A matched
    expression command without a known emotion word returns "random"; an
    unknown word returns None so the model keeps its chance to handle it.
    """
    match = _EXPRESSION_COMMAND_RE.search(transcript)
    if match is None:
        return None
    performance_noun = match.group("performance_noun")
    if performance_noun is not None:
        captured = (match.group("performance_emo") or "").strip()
        for word, intent in _EXPRESSION_COMMAND_INTENTS:
            if word in captured:
                return intent
        return _INHERENT_NOUN_INTENTS.get(performance_noun)
    captured = (match.group("emotion_cn") or "").strip() or (match.group("emotion_en") or "").strip().lower()
    if not captured:
        return "random"
    for word, intent in _EXPRESSION_COMMAND_INTENTS:
        if word in captured:
            return intent
    if captured in EMOTION_INTENTS:
        return captured
    return None


def is_direct_expression_command(transcript: str) -> bool:
    """Return True for a bare make-a-face command (做个开心的表情), False for performance requests."""
    match = _EXPRESSION_COMMAND_RE.search(transcript)
    return match is not None and match.group("performance_noun") is None


def match_spoken_emotion(text: str) -> str | None:
    """Return the intent of the first spoken emotion word, else None.

    Lexical by design: it only sees words the speaker actually says, so wordless
    sadness never matches. Negated words (不开心) are skipped, and single-char
    words (困, 累) are ignored because they match inside unrelated words
    (困难, 拖累). When several words appear, the one said first wins.
    """
    first_hit: tuple[int, str] | None = None
    for word, intent in _EXPRESSION_COMMAND_INTENTS:
        if len(word) < 2:
            continue
        index = text.find(word)
        while index != -1 and text[max(0, index - 1) : index] in _NEGATION_PREFIXES:
            index = text.find(word, index + 1)
        if index != -1 and (first_hit is None or index < first_hit[0]):
            first_hit = (index, intent)
    return first_hit[1] if first_hit is not None else None


def _normalize_emotion_key(value: str) -> str:
    """Normalize an emotion request for exact intent and keyword matching."""
    without_accents = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", without_accents.lower()).strip("_")


def _keyword_intent(normalized_key: str) -> str | None:
    """Return the first nuanced intent whose keywords are all present."""
    tokens = set(normalized_key.split("_"))
    for keywords, intent in _KEYWORD_INTENTS:
        if all(keyword in tokens for keyword in keywords):
            return intent
    return None


def resolve_emotion_name(requested_emotion: object, available_emotions: list[str]) -> str | None:
    """Resolve a compact intent, nuanced yes/no phrase, or recorded move ID."""
    if not available_emotions:
        return None

    requested = str(requested_emotion or "").strip()
    if not requested:
        return None

    normalized = _normalize_emotion_key(requested)
    if not normalized or normalized == "random":
        return None

    available_by_key = {_normalize_emotion_key(name): name for name in available_emotions}
    exact_move = available_by_key.get(normalized)
    if exact_move in _ALLOWED_MOVE_NAMES:
        return exact_move

    intent = normalized if normalized in _INTENT_TO_MOVES else None
    if intent is None:
        intent = _keyword_intent(normalized)

    if intent is None:
        return None

    for candidate in _INTENT_TO_MOVES.get(intent, ()):
        if candidate in available_emotions:
            return candidate
    return None


def random_curated_emotion(available_emotions: list[str]) -> str:
    """Choose a random emotion from the curated default pool when possible."""
    curated_available = [emotion for emotion in _CURATED_DEFAULT_MOVES if emotion in available_emotions]
    if curated_available:
        return random.choice(curated_available)
    return random.choice(available_emotions)


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a robot emotion matching a requested emotional intent."
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(EMOTION_INTENTS),
                "description": (
                    "Compact emotional intent to express. Choose one of the enum values, mapped "
                    "from the user's words: 开心/高兴→happy, 伤心/难过/悲伤→sad, 沮丧/失落→downcast, "
                    "生气/愤怒→angry, 害怕→scared, 焦虑/紧张→anxious, 惊讶→surprised, 震惊→amazed, "
                    "无聊→bored, 困→sleepy, 累→tired, 点头→yes, 摇头→no, 害羞/尴尬→embarrassed, "
                    "孤独→lonely, 喜爱→loving, 感谢→grateful, 欢迎/问候→welcoming, 再见→goodbye, "
                    "安慰→calming, 放心→relief, 不耐烦→impatient, 兴奋→excited. Use nuanced "
                    "labels like no_sad, no_excited, no_firm, or yes_understanding when plain yes/no "
                    "loses meaning. Use random if no clear intent fits."
                ),
            },
        },
        "required": [],
    }
    _library: "RecordedMoves | None" = None
    # Both the local expression trigger and the model can queue the same move in
    # one turn; the dedupe window collapses the duplicate whichever fires first.
    _last_queued_move: tuple[str, float] | None = None
    _duplicate_queue_window_s = 4.0

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        requested_emotion = kwargs.get("emotion")

        logger.info("Tool call: play_emotion emotion=%s", requested_emotion)

        try:
            if self._library is None:
                # Constructing this downloads the dataset, so it must not run at import.
                self._library = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            library = self._library
            emotion_names = library.list_moves()
            if not emotion_names:
                return {"error": "No emotions currently available"}

            emotion_name = resolve_emotion_name(requested_emotion, emotion_names)
            if not emotion_name:
                logger.info("play_emotion: %r did not resolve; using random curated", requested_emotion)
                emotion_name = random_curated_emotion(emotion_names)

            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, library)

            now = time.monotonic()
            last_queued = PlayEmotion._last_queued_move
            if (
                last_queued is not None
                and last_queued[0] == emotion_name
                and now - last_queued[1] < PlayEmotion._duplicate_queue_window_s
            ):
                logger.info(
                    "play_emotion: %s already queued %.2fs ago; skipping duplicate",
                    emotion_name,
                    now - last_queued[1],
                )
                return {"status": "already_queued", "emotion": emotion_name}
            PlayEmotion._last_queued_move = (emotion_name, now)

            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
