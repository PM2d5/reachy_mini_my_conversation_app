"""Tests for the camera-frame vision relay used by Qwen-Audio realtime models."""

from typing import Any

import pytest

from my_conversation_app import vision_relay
from my_conversation_app.config import config


class _FakeResponse:
    status_code = 200
    text = '{"choices":[{"message":{"content":"桌上有一台笔记本电脑"}}]}'

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "桌上有一台笔记本电脑"}}]}


class _FakeAsyncClient:
    last_request: dict[str, Any] = {}

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False

    async def post(
        self, url: str, headers: dict[str, str] | None = None, json: dict[str, Any] | None = None
    ) -> _FakeResponse:
        _FakeAsyncClient.last_request = {"url": url, "headers": headers, "json": json}
        return _FakeResponse()


@pytest.mark.asyncio
async def test_describe_camera_frame_sends_question_and_image(monkeypatch: Any) -> None:
    """The relay posts question + frame to the configured vision chat model."""
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(config, "DASHSCOPE_TOKEN_PLAN_API_KEY", None)
    monkeypatch.setattr(config, "DASHSCOPE_TOKEN_PLAN_CHAT_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(config, "DASHSCOPE_VISION_MODEL", "qwen3.8-flash")
    monkeypatch.setattr(vision_relay.httpx, "AsyncClient", _FakeAsyncClient)

    result = await vision_relay.describe_camera_frame("桌上有什么", "QUJD")

    assert result == {"description": "桌上有一台笔记本电脑"}
    request = _FakeAsyncClient.last_request
    assert request["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    body = request["json"]
    assert body["model"] == "qwen3.8-flash"
    content = body["messages"][0]["content"]
    assert "桌上有什么" in content[0]["text"]
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


@pytest.mark.asyncio
async def test_describe_camera_frame_prefers_token_plan(monkeypatch: Any) -> None:
    """The vision relay bills the token plan when its key and endpoint are set."""
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "payg-key")
    monkeypatch.setattr(config, "DASHSCOPE_TOKEN_PLAN_API_KEY", "plan-key")
    monkeypatch.setattr(config, "DASHSCOPE_TOKEN_PLAN_CHAT_BASE", "https://token-plan.example/v1/")
    monkeypatch.setattr(vision_relay.httpx, "AsyncClient", _FakeAsyncClient)

    await vision_relay.describe_camera_frame("桌上有什么", "QUJD")

    request = _FakeAsyncClient.last_request
    assert request["url"] == "https://token-plan.example/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer plan-key"


@pytest.mark.asyncio
async def test_describe_camera_frame_requires_api_key(monkeypatch: Any) -> None:
    """Without a DashScope key the relay returns an error instead of raising."""
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", None)
    monkeypatch.setattr(config, "DASHSCOPE_TOKEN_PLAN_API_KEY", None)

    result = await vision_relay.describe_camera_frame("桌上有什么", "QUJD")

    assert result == {"error": "vision_relay_not_configured"}
