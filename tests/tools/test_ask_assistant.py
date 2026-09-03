import json
import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from my_conversation_app.tools import ask_assistant as ask_assistant_module
from my_conversation_app.config import config
from my_conversation_app.tools.core_tools import ToolDependencies, ConversationHistory
from my_conversation_app.tools.ask_assistant import (
    BLOCKED_REPLY,
    AskAssistant,
    _merged_history,
    clean_reply_for_speech,
)


# Captured before any test patches httpx.AsyncClient, so repeated installs stack cleanly.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _make_deps() -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        conversation_history=ConversationHistory(),
    )


def _configure_openclaw(monkeypatch: pytest.MonkeyPatch, timeout_s: float = 10.0) -> None:
    monkeypatch.setattr(config, "OPENCLAW_API_URL", "http://openclaw.test/v1/chat/completions")
    monkeypatch.setattr(config, "OPENCLAW_API_TOKEN", "test-token")
    monkeypatch.setattr(config, "OPENCLAW_TIMEOUT_S", timeout_s)


def _completion(reply: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """Route the tool's HTTP client through a mock transport and capture the request."""
    captured: list[httpx.Request] = []

    async def wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        outcome = handler(request)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome

    monkeypatch.setattr(
        ask_assistant_module.httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(wrapped), **kwargs),
    )
    return captured


@pytest.mark.asyncio
async def test_success_cleans_and_records_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful call returns the cleaned reply and records it in the shared history."""
    _configure_openclaw(monkeypatch)
    captured = _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json=_completion("**需要预约**，免费入场 🎫\n- 周末票走得快"))
    )
    deps = _make_deps()
    deps.conversation_history.append("user", "安吉自然博物馆门票要预约吗")

    result = await AskAssistant()(deps, query="门票要预约吗", history=[])

    assert result == {"ok": True, "reply": "需要预约，免费入场\n周末票走得快"}
    body = json.loads(captured[0].content)
    assert body["model"] == "openclaw"
    assert body["user"] == ask_assistant_module.OPENCLAW_SESSION_USER
    assert body["messages"][-1] == {"role": "user", "content": "门票要预约吗"}
    assert body["messages"][-2] == {"role": "user", "content": "安吉自然博物馆门票要预约吗"}
    assert captured[0].headers["Authorization"] == "Bearer test-token"
    assert deps.conversation_history.snapshot()[-1] == {
        "role": "assistant",
        "content": "需要预约，免费入场\n周末票走得快",
    }


@pytest.mark.asyncio
async def test_dangerous_query_is_refused_without_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dangerous operations are refused locally and never reach the gateway."""
    _configure_openclaw(monkeypatch)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dangerous query must not be sent")

    _install_transport(monkeypatch, fail_handler)
    result = await AskAssistant()(_make_deps(), query="帮我删除电脑里的文件")

    assert result == {"ok": False, "blocked": True, "reply": BLOCKED_REPLY}


@pytest.mark.asyncio
async def test_calendar_deletion_is_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phrase-level blocking must not catch harmless calendar edits."""
    _configure_openclaw(monkeypatch)
    _install_transport(monkeypatch, lambda request: httpx.Response(200, json=_completion("已删除该日程")))

    result = await AskAssistant()(_make_deps(), query="帮我删除明天的日程")

    assert result["ok"] is True


@pytest.mark.asyncio
async def test_http_error_and_empty_reply_map_to_ok_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP failures and reply-less bodies both surface as ok=false payloads."""
    _configure_openclaw(monkeypatch)
    captured = _install_transport(monkeypatch, lambda request: httpx.Response(500, text="boom"))
    result = await AskAssistant()(_make_deps(), query="查一下新闻")
    assert result == {"ok": False, "error": "http_500"}
    assert captured

    _install_transport(monkeypatch, lambda request: httpx.Response(200, json={"choices": []}))
    result = await AskAssistant()(_make_deps(), query="查一下新闻")
    assert result == {"ok": False, "error": "empty_reply"}


@pytest.mark.asyncio
async def test_timeout_and_network_errors_map_to_ok_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts and connection failures map to the documented error keys."""
    _configure_openclaw(monkeypatch)

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    _install_transport(monkeypatch, timeout_handler)
    result = await AskAssistant()(_make_deps(), query="查一下天气")
    assert result == {"ok": False, "error": "timeout"}

    def refused_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _install_transport(monkeypatch, refused_handler)
    result = await AskAssistant()(_make_deps(), query="查一下天气")
    assert result == {"ok": False, "error": "network_error"}


@pytest.mark.asyncio
async def test_missing_query_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call without a usable query fails fast with an error dict."""
    _configure_openclaw(monkeypatch)
    assert await AskAssistant()(_make_deps()) == {"error": "query is required"}


def test_availability_requires_url_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool only reports available with a fully configured endpoint."""
    monkeypatch.setattr(config, "OPENCLAW_API_URL", "http://openclaw.test/v1/chat/completions")
    monkeypatch.setattr(config, "OPENCLAW_API_TOKEN", "token")
    assert AskAssistant().is_available() is True

    monkeypatch.setattr(config, "OPENCLAW_API_TOKEN", "")
    assert AskAssistant().is_available() is False


def test_merged_history_dedupes_and_truncates() -> None:
    """Model-supplied turns merge with the transcript without duplicates, capped per contract."""
    managed = [
        {"role": "user", "content": "我们9月11号去安吉"},
        {"role": "assistant", "content": "已建好日程：9月11-13日安吉之旅"},
    ]
    supplied = [
        {"role": "user", "content": "我们9月11号去安吉"},  # duplicate of the transcript
        {"role": "assistant", "content": "上一次的结果"},
        {"role": "bogus", "content": "ignored role"},
    ]
    merged = _merged_history(managed, supplied)
    assert merged == [
        {"role": "user", "content": "我们9月11号去安吉"},
        {"role": "assistant", "content": "已建好日程：9月11-13日安吉之旅"},
        {"role": "assistant", "content": "上一次的结果"},
    ]

    long_turns = [{"role": "user", "content": f"第{i}轮" + "长" * 300} for i in range(8)]
    truncated = _merged_history([], long_turns)
    assert len(truncated) == 5
    assert truncated[0]["content"].startswith("第3轮")
    assert all(len(turn["content"]) <= 200 for turn in truncated)


def test_clean_reply_for_speech_strips_markdown_and_emoji() -> None:
    """Markdown markers and emoji never reach the spoken relay."""
    raw = (
        "# 攻略\n\n"
        "**门票**：免费，*需预约*\n"
        "- 周末票走得快\n"
        "详见[官网](https://example.com)和![地图](https://example.com/map.png)\n"
        "`代码块` 和~~删除线~~ 🎫✨"
    )
    cleaned = clean_reply_for_speech(raw)
    assert cleaned == ("攻略\n\n门票：免费，需预约\n周末票走得快\n详见官网和\n代码块 和删除线")


def test_clean_reply_caps_length() -> None:
    """A runaway reply is capped before it can flood the relay turn."""
    assert len(clean_reply_for_speech("长" * 5000)) == ask_assistant_module.MAX_REPLY_CHARS


def test_conversation_history_bounds_and_clears_per_session() -> None:
    """History keeps the last turns, drops empty text, and clears on a new session."""
    history = ConversationHistory(max_turns=3)
    history.append("user", "")
    history.append("user", "  ")
    for index in range(5):
        history.append("user", f"第{index}句")

    assert [turn["content"] for turn in history.snapshot()] == ["第2句", "第3句", "第4句"]

    history.new_session()
    assert history.snapshot() == []


def test_schema_stays_query_only() -> None:
    """History must stay out of the schema: Qwen truncates long call arguments."""
    assert set(AskAssistant.parameters_schema["properties"]) == {"query"}
