"""Camera-frame captioning for text-only realtime models.

The Qwen-Audio realtime family understands speech but not images, so the
camera tool's frame is described by a multimodal DashScope chat model and the
description is relayed to the realtime model as text. Omni realtime models see
the image itself and never go through this path.
"""

import logging
from typing import Any

import httpx

from my_conversation_app.config import config, dashscope_vision_credentials
from my_conversation_app.tools.ask_assistant import extract_reply


logger = logging.getLogger(__name__)

VISION_RELAY_TIMEOUT_S = 20.0

VISION_RELAY_PROMPT = (
    "你是机器人的眼睛。根据摄像头拍到的画面，用中文简明准确地回答问题；只描述画面里真实可见的内容，不要猜测。"
)


async def describe_camera_frame(question: str, b64_jpeg: str) -> dict[str, Any]:
    """Answer the camera question from the frame via a vision chat model."""
    api_key, chat_completions_url = dashscope_vision_credentials()
    if not api_key:
        logger.warning("Vision relay called but no DashScope API key is configured")
        return {"error": "vision_relay_not_configured"}

    if not isinstance(b64_jpeg, str):
        logger.warning("Unexpected type for camera frame: %s", type(b64_jpeg))
        b64_jpeg = str(b64_jpeg)

    async with httpx.AsyncClient(timeout=VISION_RELAY_TIMEOUT_S) as client:
        try:
            response = await client.post(
                chat_completions_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": config.DASHSCOPE_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{VISION_RELAY_PROMPT}\n问题：{question}"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}},
                            ],
                        }
                    ],
                },
            )
        except httpx.TimeoutException:
            logger.warning("Vision relay timed out after %ss", VISION_RELAY_TIMEOUT_S)
            return {"error": "timeout"}
        except httpx.HTTPError as exc:
            logger.warning("Vision relay request failed: %s", exc)
            return {"error": "network_error"}

    if response.status_code != 200:
        logger.warning("Vision relay returned HTTP %d: %s", response.status_code, response.text[:200])
        return {"error": f"http_{response.status_code}"}

    description = extract_reply(response.json()).strip()
    if not description:
        logger.warning("Vision relay response had no reply content: %s", response.text[:200])
        return {"error": "empty_reply"}
    logger.info("Vision relay described camera frame (%d chars)", len(description))
    return {"description": description}
