import logging
from unittest.mock import MagicMock

import pytest

from my_conversation_app.tools import play_emotion as play_emotion_module
from my_conversation_app.tools.core_tools import ToolDependencies
from my_conversation_app.tools.play_emotion import (
    EMOTION_INTENTS,
    PlayEmotion,
    match_spoken_emotion,
    resolve_emotion_name,
    random_curated_emotion,
    match_expression_command,
    is_direct_expression_command,
)


@pytest.fixture(autouse=True)
def _reset_queued_emotion() -> None:
    """Keep the cross-test dedupe window from leaking between tests."""
    PlayEmotion._last_queued_move = None


AVAILABLE_EMOTIONS = [
    "cheerful1",
    "confused1",
    "no1",
    "no_sad1",
    "no_excited1",
    "resigned1",
    "understanding2",
    "yes_sad1",
]


def test_play_emotion_schema_uses_compact_intents() -> None:
    """Expose compact intents instead of the full recorded-move catalog."""
    emotion_schema = PlayEmotion.parameters_schema["properties"]["emotion"]

    assert emotion_schema["enum"] == list(EMOTION_INTENTS)
    assert "no_sad" in emotion_schema["enum"]
    assert "no_excited" in emotion_schema["enum"]
    assert "no_firm" in emotion_schema["enum"]
    assert "yes_understanding" in emotion_schema["enum"]
    assert "no_confused" not in emotion_schema["enum"]
    assert "oops" not in emotion_schema["enum"]
    assert "yes_sad" not in emotion_schema["enum"]
    assert "yes_proud" not in emotion_schema["enum"]
    assert "loving1" not in emotion_schema["enum"]
    assert "Available emotions" not in emotion_schema["description"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("no_sad1", "no_sad1"),
        ("sad no", "no_sad1"),
        ("no_excited", "no_excited1"),
        ("yes_understanding", "understanding2"),
    ],
)
def test_resolve_emotion_name_accepts_ids_intents_and_yes_no_phrases(requested: str, expected: str) -> None:
    """Resolve exact IDs, compact intents, and exposed yes/no phrase variants."""
    assert resolve_emotion_name(requested, AVAILABLE_EMOTIONS) == expected


def test_resolve_emotion_name_returns_none_for_random_or_unknown() -> None:
    """Let the caller choose a random fallback when there is no resolved match."""
    assert resolve_emotion_name("random", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("contento", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("totally mysterious mood", AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    "removed_intent",
    [
        "confused no",
        "curious",
        "inquiring",
        "lost",
        "no_confused",
        "oops",
        "proud",
        "uncomfortable",
        "yes proud",
        "yes sad",
        "yes_proud",
        "yes_sad",
    ],
)
def test_resolve_emotion_name_does_not_accept_removed_substitute_intents(removed_intent: str) -> None:
    """Removed intents should not resolve through unrelated substitute moves."""
    assert resolve_emotion_name(removed_intent, AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    ("intent", "poor_options"),
    [
        ("excited", ["success2"]),
        ("grateful", ["helpful1", "loving1"]),
        ("happy", ["loving1"]),
        ("lonely", ["sad1"]),
        ("no", ["no_sad1", "no_excited1"]),
        ("no_excited", ["no1"]),
        ("no_sad", ["downcast1"]),
        ("uncertain", ["resigned1"]),
        ("yes_understanding", ["yes1"]),
    ],
)
def test_resolve_emotion_name_does_not_use_weak_fallbacks(intent: str, poor_options: list[str]) -> None:
    """Do not use loosely related moves when a precise move is unavailable."""
    assert resolve_emotion_name(intent, poor_options) is None


@pytest.mark.parametrize("bad_move", ["cheerful1", "oops1", "oops2", "reprimand3", "understanding1", "yes_sad1"])
def test_resolve_emotion_name_does_not_accept_bad_exact_moves(bad_move: str) -> None:
    """Bad-quality recorded move IDs should not bypass the curated resolver."""
    assert resolve_emotion_name(bad_move, [*AVAILABLE_EMOTIONS, bad_move]) is None


@pytest.mark.parametrize(
    "ambiguous_move",
    [
        "contempt1",
        "curious1",
        "dance1",
        "furious1",
        "helpful2",
        "impatient1",
        "incomprehensible2",
        "inquiring1",
        "lost1",
        "proud2",
        "proud3",
        "tired1",
        "uncomfortable1",
        "welcoming1",
    ],
)
def test_resolve_emotion_name_does_not_accept_redundant_ambiguous_exact_moves(ambiguous_move: str) -> None:
    """OK ambiguous moves should be skipped when clear or excellent alternatives exist."""
    assert resolve_emotion_name(ambiguous_move, [*AVAILABLE_EMOTIONS, ambiguous_move]) is None


def test_random_curated_emotion_uses_curated_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random fallback should avoid non-curated moves when curated options exist."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1", "yes_sad1", "confused1"]) == "confused1"
    assert choices_seen == ["confused1"]


def test_random_curated_emotion_falls_back_when_no_curated_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should still return an available move if the curated pool is unavailable."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1"]) == "cheerful1"
    assert choices_seen == ["cheerful1"]


@pytest.mark.asyncio
async def test_play_emotion_queues_resolved_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should queue the resolved recorded-move ID."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())
    result = await tool(deps, emotion="sad no")

    assert result == {"status": "queued", "emotion": "no_sad1"}
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "no_sad1"


@pytest.mark.asyncio
async def test_play_emotion_queues_random_for_unknown_emotion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown explicit values should fall back to a random recorded emotion."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    def fake_choice(emotion_names: list[str]) -> str:
        assert "cheerful1" not in emotion_names
        assert "yes_sad1" not in emotion_names
        return "confused1"

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())
    with caplog.at_level(logging.INFO, logger=play_emotion_module.logger.name):
        result = await tool(deps, emotion="contento")

    assert result == {"status": "queued", "emotion": "confused1"}
    assert "play_emotion: 'contento' did not resolve; using random curated" in caplog.text
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "confused1"


@pytest.mark.parametrize(
    ("transcript", "expected_intent"),
    [
        ("做一个伤心的表情。", "sad"),
        ("做个伤心的表情。", "sad"),
        ("弄一个伤心的表情。", "sad"),
        ("来个开心的表情。", "happy"),
        ("你表演一个害怕的情绪。", "scared"),
        ("做个尴尬的表情", "embarrassed"),
        ("做一个同意的表情。", "yes"),
        ("做个不同意的表情。", "no"),
        ("做个表情。", "random"),
        ("来一个表情", "random"),
        ("Can you do a sad emotion?", "sad"),
        ("please do an angry face", "angry"),
        ("Show me a happy expression", "happy"),
    ],
)
def test_match_expression_command_matches_explicit_commands(transcript: str, expected_intent: str) -> None:
    """Explicit show-an-expression commands resolve to their intent."""
    assert match_expression_command(transcript) == expected_intent


@pytest.mark.parametrize(
    "transcript",
    [
        "我今天很伤心。",
        "跟我讲讲你伤心的事。",
        "你现在什么情绪？",
        "这个表情怎么做？",
        "再见。",
        "帮我看看左边有什么。",
        "do you have emotions?",
    ],
)
def test_match_expression_command_ignores_non_commands(transcript: str) -> None:
    """Emotional small talk and questions must not trigger the local move."""
    assert match_expression_command(transcript) is None


@pytest.mark.asyncio
async def test_play_emotion_skips_duplicate_move_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local trigger and a model call for the same move queue it only once."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    tool = PlayEmotion()
    monkeypatch.setattr(tool, "_library", FakeRecordedMoves())

    first = await tool(deps, emotion="no_sad")
    duplicate = await tool(deps, emotion="no_sad")

    assert first == {"status": "queued", "emotion": "no_sad1"}
    assert duplicate == {"status": "already_queued", "emotion": "no_sad1"}
    assert movement_manager.queue_move.call_count == 1


@pytest.mark.parametrize(
    ("transcript", "expected_intent"),
    [
        ("讲一个悲伤的故事。", "sad"),
        ("讲个伤心的故事。", "sad"),
        ("说个恐怖故事。", "scared"),
        ("唱首开心的歌。", "happy"),
        ("讲个笑话。", "happy"),
        ("来个搞笑的段子。", "happy"),
    ],
)
def test_match_expression_command_matches_performance_requests(transcript: str, expected_intent: str) -> None:
    """Mood-named performance requests (讲个悲伤的故事) resolve to their intent."""
    assert match_expression_command(transcript) == expected_intent


@pytest.mark.parametrize(
    "transcript",
    [
        ("给我讲个故事。"),
        ("讲个故事吧。"),
        ("这本书讲了一个很长很长的故事。"),
    ],
)
def test_match_expression_command_ignores_moodless_performance_requests(transcript: str) -> None:
    """A story request with no named mood stays with the model layer."""
    assert match_expression_command(transcript) is None


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("做一个伤心的表情。", True),
        ("做个表情。", True),
        ("来一个搞笑的动作", True),
        ("Can you do a sad face?", True),
        ("讲一个开心的故事。", False),
        ("唱首伤心的歌。", False),
        ("讲个笑话。", False),
        ("我今天很伤心。", False),
        ("跟我讲讲你伤心的事。", False),
    ],
)
def test_is_direct_expression_command_splits_bare_commands_from_performances(transcript: str, expected: bool) -> None:
    """Only bare make-a-face commands count; performance requests and small talk do not."""
    assert is_direct_expression_command(transcript) is expected


@pytest.mark.parametrize(
    ("spoken_text", "expected_intent"),
    [
        ("从前有一只小狐狸，它总是很孤独", "lonely"),
        ("它很孤独，后来它走了，大家都很难过", "lonely"),
        ("小狐狸很难过，因为它知道它们再也见不到了", "sad"),
        ("这个消息真是太让人兴奋了！", "excited"),
        ("听到这个消息我特别惊讶", "surprised"),
        ("你别再难过了，都会好起来的。", "sad"),
        ("我不同意这个做法。", "no"),
        ("我同意你的看法。", "yes"),
    ],
)
def test_match_spoken_emotion_hits_narrated_emotion_words(spoken_text: str, expected_intent: str) -> None:
    """Emotion words in the model's own speech resolve to their intent.

    Comforting phrases (别难过) count: the robot showing concern while it
    comforts is the desired empathy.
    """
    assert match_spoken_emotion(spoken_text) == expected_intent


@pytest.mark.parametrize(
    "spoken_text",
    [
        ("小狐狸每天在河边等，等到冬天来了，小鸟再也没有回来。"),
        ("今天天气不错，我们聊聊别的吧。"),
        ("我很不开心。"),
        ("这个任务很困难。"),
    ],
)
def test_match_spoken_emotion_ignores_wordless_or_negated_text(spoken_text: str) -> None:
    """Wordless narration, negations, and single-char false friends stay silent."""
    assert match_spoken_emotion(spoken_text) is None
