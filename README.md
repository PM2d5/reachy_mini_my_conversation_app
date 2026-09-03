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

## Feature inventory

[`docs/features.md`](docs/features.md) is the always-current inventory of everything this app does:
conversation backends, wake word and idle policy, every LLM tool, the profile system,
the web UI and its JSON-RPC API, and all configuration variables. Update it in the same
PR whenever a feature changes (see `AGENTS.md`).

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

## Wake word listening

By default (`REACHY_MINI_WAKE_WORD_ENABLED=1`) the app does not listen continuously.
The first session behaves as before: Reachy greets you and listens. Once you say a
goodbye ("再见", "拜拜", "goodbye", configurable) or stay silent for 5 minutes, the
realtime session pauses, Reachy retracts its neck while keeping the head level
(distinct from the sleep pose) and holds its antennas still, and the mic only
feeds an offline wake word detector (openWakeWord). Saying the wake word lifts
the head back up and resumes the session with a brief spoken acknowledgement — just a
couple of words, like "I'm here", not a full greeting. Each wake opens a memoryless
session, so the app rotates between a few acknowledgement flavors to keep them varied.

The default wake word is **"hi reachy"**, detected by a custom model bundled in the
app (`src/my_conversation_app/audio/models/hi_reachy.onnx`), trained locally with
openWakeWord's automated pipeline. Set `REACHY_MINI_WAKE_WORD_MODELS` to use other
pretrained models ("hey mycroft", "alexa", ...) or custom `.onnx` model paths instead.

```bash
# REACHY_MINI_WAKE_WORD_ENABLED=1                          # 0 keeps always-on listening
# REACHY_MINI_WAKE_WORD_MODELS=hey_mycroft                 # comma-separated openWakeWord model names or .onnx/.tflite paths
# REACHY_MINI_WAKE_WORD_THRESHOLD=0.5                      # 0..1, lower = more sensitive
# REACHY_MINI_WAKE_WORD_ACTIVE_TIMEOUT_S=300               # idle exit delay, 0 disables
# REACHY_MINI_GOODBYE_KEYWORDS=再见,拜拜,goodbye,bye-bye,bye bye
```

Pretrained models (hey_mycroft, hey_jarvis, alexa, ...) are downloaded automatically
on first use. openWakeWord ships no Chinese pretrained models: to use a Chinese wake
word, train a custom model (see the openWakeWord docs) and point
`REACHY_MINI_WAKE_WORD_MODELS` at the model file. If no model can be loaded, the app
logs a warning and falls back to always-on listening.


Do not forget to customize:
- this `README.md` file
- the `index.html` file (Hugging Face Spaces landing page)
- the `src/my_conversation_app/static/index.html` (the web app parameters page)

The original README from the conversation app is available in `README_OLD.md`.