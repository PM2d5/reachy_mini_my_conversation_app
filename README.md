---
title: My Conversation App
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# My Conversation App

Forked from the Reachy Mini conversation app.

Customize `profiles/_my_conversation_app_locked_profile/profile.md` to change the assistant instructions and enabled tools.
Add custom tools under `src/my_conversation_app/tools/` by subclassing `Tool`.

## Chinese realtime backend (DashScope)

This fork can swap the default Hugging Face realtime backend for Alibaba
DashScope's Qwen-Omni-Realtime, which transcribes and speaks Chinese natively.
Set the following in `.env`:

```bash
REALTIME_BACKEND="dashscope"
DASHSCOPE_API_KEY=sk-...            # required; from Alibaba Model Studio
# DASHSCOPE_REALTIME_MODEL="qwen3.5-omni-flash-realtime"
# DASHSCOPE_REALTIME_WS_BASE="wss://dashscope.aliyuncs.com/api-ws/v1"
# DASHSCOPE_REALTIME_VOICE="Tina"      # 55+ voices: see the DashScope voice list
```

Leave `REALTIME_BACKEND` unset (or `huggingface`) to keep the default backend.
See `.env.example` for the full list.

Do not forget to customize:
- this `README.md` file
- the `index.html` file (Hugging Face Spaces landing page)
- the `src/my_conversation_app/static/index.html` (the web app parameters page)

The original README from the conversation app is available in `README_OLD.md`.