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

Do not forget to customize:
- this `README.md` file
- the `index.html` file (Hugging Face Spaces landing page)
- the `src/my_conversation_app/static/index.html` (the web app parameters page)

The original README from the conversation app is available in `README_OLD.md`.