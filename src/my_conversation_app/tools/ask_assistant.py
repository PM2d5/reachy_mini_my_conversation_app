import re
import logging
from typing import Any, Dict, List

import httpx

from my_conversation_app.config import config
from my_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# The doc contract caps context at 5 turns of 200 chars each.
MAX_HISTORY_TURNS = 5
MAX_HISTORY_CHARS_PER_TURN = 200
# Safety valve so a runaway assistant reply cannot flood the relay turn.
MAX_REPLY_CHARS = 1200

BLOCKED_REPLY = "这个操作有风险，请到微信上跟助手说。"

# Stable OpenClaw session key: the assistant keeps one continuous session across
# Reachy wakes and restarts, so it remembers earlier asks without our help.
OPENCLAW_SESSION_USER = "reachy-mini"

# Phrase-level patterns: bare "删除" would also catch harmless calendar edits.
DANGEROUS_PATTERNS = (
    "删除文件",
    "删掉文件",
    "删除照片",
    "删除电脑",
    "清空",
    "格式化",
    "卸载",
    "转账",
    "汇款",
    "付款",
    "支付",
    "下单",
    "购买",
    "发消息",
    "发短信",
    "发微信",
    "发邮件",
)

_EMOJI_PATTERN = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff\ufe0f\u200d]+")


def clean_reply_for_speech(text: str) -> str:
    """Strip markdown markup and emoji so the reply can be relayed as spoken text."""
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)  # links keep their label
    cleaned = re.sub(r"(\*\*\*|\*\*|\*|__|~~|`)+", "", cleaned)  # emphasis and code markers
    cleaned = re.sub(r"^\s{0,3}(#{1,6}|>)\s*", "", cleaned, flags=re.MULTILINE)  # headings, quotes
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)  # bullet markers
    cleaned = _EMOJI_PATTERN.sub("", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()[:MAX_REPLY_CHARS]


def _truncate_history(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Cap history to the last MAX_HISTORY_TURNS turns of MAX_HISTORY_CHARS_PER_TURN chars."""
    return [
        {"role": turn["role"], "content": turn["content"][:MAX_HISTORY_CHARS_PER_TURN]}
        for turn in turns[-MAX_HISTORY_TURNS:]
    ]


def _merged_history(managed: List[Dict[str, str]], supplied: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Dedupe model-supplied turns against the transcript, then truncate."""
    seen = {(turn["role"], turn["content"]) for turn in managed}
    merged = list(managed)
    for turn in supplied:
        key = (turn.get("role", ""), turn.get("content", ""))
        if turn.get("role") in ("user", "assistant") and turn.get("content") and key not in seen:
            seen.add(key)
            merged.append({"role": turn["role"], "content": turn["content"]})
    return _truncate_history(merged)


def _extract_reply(payload: object) -> str:
    """Pull the assistant message text out of an OpenAI-style chat completion body."""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


class AskAssistant(Tool):
    """Delegate a complex task to the OpenClaw home assistant."""

    name = "ask_assistant"
    description = (
        "调用家庭助手（OpenClaw）处理复杂任务。必须在以下场景调用："
        "1) 任何涉及实时信息的问题（价格、门票、营业时间、天气、新闻、库存）；"
        "2) 需要执行操作（建日程、设提醒、记录信息）；"
        "3) 涉及用户个人记忆、历史、家庭信息的问题；"
        "4) 多步推理、计算、资料整理；"
        "5) 自己不确定答案时。"
        "宁可多调，禁止凭训练知识编造实时信息，搜不到就明说搜不到。"
        "按信息类型判断是否调用，不看用户措辞。"
        "本工具只处理查询、问答、记录类任务；用户要求删除文件、卸载软件、发消息、花钱等危险操作时不要调用。"
        "只需传 query 参数：把当前问题完整写进 query 即可，最近对话（含上一次结果）由系统自动带给助手。"
    )
    silence_user_audio_while_running = True
    # History must stay out of the schema: Qwen truncates long function-call
    # arguments, and the server-side transcript already carries it to OpenClaw.
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "当前用户请求的完整描述",
            },
        },
        "required": ["query"],
    }

    def is_available(self) -> bool:
        """Require a configured OpenClaw endpoint and token."""
        return bool(config.OPENCLAW_API_URL) and bool((config.OPENCLAW_API_TOKEN or "").strip())

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Send the query to OpenClaw and return its cleaned reply for relaying."""
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "query is required"}

        if any(pattern in query for pattern in DANGEROUS_PATTERNS):
            logger.info("ask_assistant blocked dangerous query: %s", query)
            return {"ok": False, "blocked": True, "reply": BLOCKED_REPLY}

        if not self.is_available():
            logger.warning("ask_assistant called but OpenClaw is not configured")
            return {"ok": False, "error": "not_configured"}

        managed_history = deps.conversation_history.snapshot() if deps.conversation_history else []
        supplied_history = [turn for turn in kwargs.get("history") or [] if isinstance(turn, dict)]
        history = _merged_history(managed_history, supplied_history)

        messages = [*history, {"role": "user", "content": query.strip()}]
        logger.info("ask_assistant query=%r history_turns=%d", query, len(history))

        try:
            async with httpx.AsyncClient(timeout=config.OPENCLAW_TIMEOUT_S) as client:
                response = await client.post(
                    config.OPENCLAW_API_URL,
                    headers={"Authorization": f"Bearer {config.OPENCLAW_API_TOKEN}"},
                    json={"model": "openclaw", "user": OPENCLAW_SESSION_USER, "messages": messages},
                )
        except httpx.TimeoutException:
            logger.warning("ask_assistant timed out after %ss", config.OPENCLAW_TIMEOUT_S)
            return {"ok": False, "error": "timeout"}
        except httpx.HTTPError as exc:
            logger.warning("ask_assistant request failed: %s", exc)
            return {"ok": False, "error": "network_error"}

        if response.status_code != 200:
            logger.warning("ask_assistant returned HTTP %d: %s", response.status_code, response.text[:200])
            return {"ok": False, "error": f"http_{response.status_code}"}

        payload: object = response.json()
        reply = _extract_reply(payload)
        if not reply.strip():
            logger.warning("ask_assistant response had no reply content: %s", str(payload)[:200])
            return {"ok": False, "error": "empty_reply"}

        cleaned = clean_reply_for_speech(reply)
        if deps.conversation_history:
            deps.conversation_history.append("assistant", cleaned)
        logger.info("ask_assistant reply (%d chars)", len(cleaned))
        return {"ok": True, "reply": cleaned}
