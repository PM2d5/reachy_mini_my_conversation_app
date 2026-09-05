# 功能清单（Feature Inventory）

> 本文件是 my_conversation_app 当前功能的权威清单。**任何改变应用行为的改动，必须在同一个 PR 里更新本文件**（规则见 `AGENTS.md` 的 *Documentation* 一节）。
>
> 最后核对：2026-09-05 · `master`（含 ask_assistant / OpenClaw 对接、天线等待摆动、DashScope 摄像头图像注入、视觉规则指令与会话温度配置）

应用运行在 Reachy Mini SDK（`reachy_mini`）之上：语音进、语音出 + 机器人动作的实时对话应用，带 Web 管理界面、人格系统、长期记忆、可扩展的 LLM 工具体系（含远程 MCP Tool Spaces）。架构图见 `README.md`（源文件 `docs/scheme.mmd`）。

## 目录

1. [启动与运行](#1-启动与运行)
2. [实时语音对话](#2-实时语音对话)
3. [实时后端](#3-实时后端)
4. [唤醒词与休眠策略](#4-唤醒词与休眠策略)
5. [机器人动作系统](#5-机器人动作系统)
6. [LLM 工具清单](#6-llm-工具清单)
7. [后台任务管理](#7-后台任务管理)
8. [Tool Spaces（远程 MCP 工具）](#8-tool-spaces远程-mcp-工具)
9. [人格（Profile）系统](#9-人格profile系统)
10. [长期记忆](#10-长期记忆)
11. [Web UI 与控制接口](#11-web-ui-与控制接口)
12. [配置参考](#12-配置参考)
13. [本文档的维护规则](#13-本文档的维护规则)

---

## 1. 启动与运行

代码：`main.py`、`app_lifecycle.py`、`utils.py`、`console.py`

**入口**

- CLI：`my-conversation-app`（`main.py:main`）
- Reachy Mini 桌面应用插件：entry point `reachy_mini_apps` → `MyConversationApp`（`main.py`），Web 界面地址 `http://0.0.0.0:7860/`

**CLI 参数**（`utils.py:parse_args`）

| 参数 | 作用 |
|---|---|
| `--no-camera` | 禁用摄像头（`camera` 工具返回错误） |
| `--ui` | 额外启动 Web UI（uvicorn，`0.0.0.0:7860`） |
| `--debug` | DEBUG 日志级别 |
| `--robot-name NAME` | 与 daemon 的 `--robot-name` 匹配 |
| 子命令 `tool-spaces add <slug> [--install-only] [--profile P]` / `remove <slug>` / `list` | 管理 Tool Spaces（见 §8） |

**启动流程**（按顺序）

1. 连接 Reachy Mini daemon；失败则打印 5 步排障指引并退出（无离线模拟模式）。
2. 若机器人处于睡眠姿态（SDK `SLEEP_HEAD_POSE` 容差内）→ `enable_motors()` + `wake_up()`。
3. 迁移旧版 profile 副文件并初始化工具注册表；所选 profile 加载失败时回退到 `default`（除非被 `LOCKED_PROFILE` 锁定）。
4. 启动 `MovementManager`（呼吸等空闲动作）与 `robot.enable_wobbling()`。
5. 加载实例 `.env`；若实时后端未配置（无 API key 等）进入 `waiting_for_config` 状态，轮询等待用户在 Web 设置页完成配置——期间 Web UI 始终可用。
6. 启动麦克风采集/播放循环；对 XVF3800 DSP 应用启动音频参数（`audio/startup_config.py`：AGC/NS 等调优值）。
7. 启动应用级无活动守护线程：超时后执行 `go_to_sleep`（见 §4）。

**关闭**

- 外部停止（dashboard/手机）：active 状态下只停应用，机器人保持清醒，由 daemon 归位；standby 状态下应用退出前先从待机位姿直接进入睡眠位姿（避免 daemon 先归零、再入睡的抬头又低头动作）。
- 睡眠路径（`go_to_sleep` 工具或应用级超时）：停抖动、停动作管理器（不归位）→ 从当前位姿直接插值到 SDK 睡眠位姿（`goto_sleep_from_current_pose`，不绕行中立位，避免待机缩头时先抬头再低头）→ 请求停止当前应用。下次启动时由第 2 步唤醒。

## 2. 实时语音对话

代码：`console.py`、`huggingface_realtime.py`、`streaming.py`

- **开场问候**：每次会话建立后注入 `greeting` 提示（profile 可自定义；默认指令要求用一句符合人设的话自然开场）；从唤醒词恢复时改为极简应答（见第 4 节）。
- **聆听与打断**：服务端 VAD 负责断句；用户一开口（`speech_started`）即本地清空播放队列实现真打断（barge-in），同时冻结天线动作表示"在听"。
- **转写流**：用户部分/最终转写、助手转写均推送到控制台日志与 JSON-RPC 客户端（`conversation.transcript` 通知）。
- **回合状态机**：对外广播 `listening / thinking / speaking / ready`（`conversation.turn` 通知），驱动 UI 光球。
- **工具调用**：模型发起的函数调用全部后台执行（见 §7）；结果回填为 `function_call_output`，仅在工具出错或 `needs_response=True` 时触发一次语音跟进；`camera` 拍到的 JPEG 会作为图片消息重新注入多模态对话。所有人格的会话指令以「视觉规则」开头（`prompts.py` 的 `CAMERA_TOOL_RULE`，含中文触发词与少样本示例）：camera 工具就是模型的眼睛，视觉请求必须先调工具、只依据照片作答，禁止凭空描述或谎称没有摄像头——多模态实时模型（如 Qwen-Omni）否则会直接幻觉作答或拒答而不调工具；该规则置于指令最前（实测置顶 5/5、置尾 3/5 命中工具调用）。
- **家庭助手等待态**：`ask_assistant`（见 §6）在飞期间，麦克风音频被整体丢弃（服务端 VAD 收不到任何输入，家人闲聊绝不误触发回合），并清空残留输入缓冲；等待期间活动空闲退出（§4）被挂起，结果回来即恢复。工具启动瞬间由应用代播一句垫话——风格池确定性轮换 + 模型自然发挥（机制同唤醒应答 `ASSISTANT_WAIT_ACKNOWLEDGEMENT_PROMPTS`），人设指令明确要求模型不自行播报垫话；垫话以用户消息注入后残留在上下文中，结果回来时紧跟 `function_call_output` 再注入一条转述锚点指令（`ASSISTANT_RESULT_RELAY_PROMPT`），防止模型重复垫话而不转述结果。新会话（含唤醒恢复）会重置滚动对话历史；OpenClaw 侧使用固定会话标识 `reachy-mini`，跨唤醒/重启保持同一个助手会话（靠 OpenClaw 自身的上下文管理记忆早前询问）。请求在飞期间两根天线以进入时的姿态为中心同向左右摆动（幅度 20°、0.6 Hz，`MovementManager.set_busy_sway`）作为"思考中"的可视提示，拿到结果（含超时/网络错误）后经 0.4s 混合平滑归位。
- **错误恢复**：后端连接失败按指数退避重试 3 次，外层每 5 秒重连，期间 Web UI 保持可用；单次会话内一个时间只有一条活跃回复（对模型的并发 `response.create` 做串行合并与重试）。

## 3. 实时后端

代码：`huggingface_realtime.py`、`dashscope_realtime.py`、`conversation_handler.py`（抽象基类）

用 `REALTIME_BACKEND` 选择，两个后端共享同一会话循环与工具协议：

| | Hugging Face（默认） | DashScope（`dashscope`） |
|---|---|---|
| 模型 | 部署在 HF Space 上的后端（Qwen3-TTS CustomVoice） | Qwen-Omni-Realtime，默认 `qwen3.5-omni-flash-realtime` |
| 中文 | 输入转写语言可设 `REALTIME_TRANSCRIPTION_LANGUAGE=zh`（默认 `en`，转写模型 `gpt-4o-transcribe`） | 原生多语言 ASR + 中文语音，无需转写配置 |
| 语音 | 9 个：Aiden（默认）、Ryan、Dylan、Eric、Ono_Anna、Serena、Sohee、Uncle_Fu、Vivian | 56 个（Tina 默认），完整列表见阿里云文档 |
| 连接 | `deployed`（默认，经 session proxy，支持 `HF_TOKEN` 鉴权）或 `local`（直连 `HF_REALTIME_WS_URL`，如局域网 `ws://host:8765/v1/realtime`） | `wss://dashscope.aliyuncs.com/api-ws/v1`，需 `DASHSCOPE_API_KEY` |
| 音频 | 16 kHz PCM 原生速率直传 | 输出 24 kHz PCM，客户端线性重采样到 16 kHz；长 MCP 工具名自动改写为短别名 |
| 视觉 | `camera` 拍照以 `input_image` 消息注入对话 | 连接层把 `input_image` 消息翻译为 `input_image_buffer.append`（纯 base64，单帧/单图 ≤256 KB；超限自动降到 720p/更低质量重编码，仍超限则丢弃并记 error）+ `input_audio_buffer.commit` 提交进对话；因服务端 VAD 只随语音提交图片，注入瞬间短暂切到手动断句并附 1 s 合成噪声（仅上送服务器，不外放），随后恢复原断句配置 |

语音切换和人格切换均为在线热更新（`session.update` + 会话重建，无需重启应用）。

## 4. 唤醒词与休眠策略

代码：`audio/wake_word.py`、`console.py`、`idle_policy.py`、`config.py`

会话有两个相位，经 `conversation.phase {phase, reason}` 广播：

- **active**：正常聆听对话。首次启动即进入，机器人先问候。
- **standby**：实时会话暂停，机器人缩头下沉（头部保持水平、不低头，与真正的休眠姿态区分；天线垂落静止，2 s 插值），期间抑制呼吸动作；麦克风只喂给离线唤醒词检测器（openWakeWord，ONNX，16 kHz）。默认唤醒词 **"hi reachy"**，由内置自训模型 `audio/models/hi_reachy.onnx` 检测；说出唤醒词即抬头回中立位（1.5 s 插值）并恢复会话，以极简应答代替重新问候（`WAKE_ACKNOWLEDGEMENT_PROMPTS`：两三个词、跟所讲语言一致，如"我在""干嘛"；每次唤醒是无记忆的新会话，由应用在几种应答风格间轮换并随机起步，保证说法有变化）。

进入 standby 的两个触发条件：

1. **告别关键词**：最终用户转写中包含 `REACHY_MINI_GOODBYE_KEYWORDS` 之一（默认 `再见, 拜拜, goodbye, bye-bye, bye bye`），reason=`goodbye_keyword`。
2. **活动空闲超时**：`REACHY_MINI_WAKE_WORD_ACTIVE_TIMEOUT_S`（默认 300 秒，0 禁用）无对话活动，reason=`idle_timeout`。

其他相关行为：

- **会话内空闲小动作**（`idle_policy.py`）：active 状态下静默 3 分钟且无进行中的回复时，本地按权重随机执行一个动作工具（不经 LLM）：`idle_do_nothing` 0.60 / `dance` 0.16 / `play_emotion` 0.16 / `move_head`（随机方向）0.08。
- **应用级休眠**：`REACHY_MINI_APP_TIMEOUT_MINUTES`（默认 1440 分钟）无活动 → 机器人入睡并停止应用；LLM 也可主动调用 `go_to_sleep` 工具。
- **降级**：唤醒词模型加载失败时记录警告并退回常开聆听；可用 `REACHY_MINI_WAKE_WORD_DUMP_DIR` 转储待机麦克风音频排查误检/漏检。

## 5. 机器人动作系统

代码：`moves.py`、`dance_emotion_moves.py`

- `MovementManager`：60 Hz 控制循环，串行化所有动作命令；聆听时冻结天线（0.15 s 防抖、0.4 s 淡回）；说话时暂停人脸跟踪并锚定视线；空闲 0.3 s 后自动开始呼吸动作（5 mm 起伏 + 天线慢摆）；唤醒词待机时头部水平下沉并保持静止（呼吸被抑制），唤醒时回中立位。
- `BreathingMove`（呼吸）、`DanceQueueMove`（舞蹈库动作）、`EmotionQueueMove`（情绪库预录动作，可叠加在跟踪锚点上）、`GotoQueueMove`（头/天线/身体偏航线性插值）。
- 动作库：`reachy_mini_dances_library`（20 支命名舞蹈）与 `RecordedMoves("pollen-robotics/reachy-mini-emotions-library")`（首次使用时下载）。

## 6. LLM 工具清单

代码：`tools/`（每个工具一个文件，契约见 `tools/core_tools.py`）

所有工具异步执行、返回 dict、失败返回 `{"error": ...}` 而不抛入会话循环。每个 profile 经 frontmatter `default_tools` + 实例覆盖（§9）决定启用的工具子集；系统工具始终启用。

### 内置机器人工具

| 工具 | 参数 | 作用 |
|---|---|---|
| `camera` | `question`（必填） | 拍一帧 JPEG 注入对话让模型"看"眼前事物（需未传 `--no-camera`） |
| `dance` | `move`（20 支可选，省略=随机）、`repeat`（默认 1） | 队列播放舞蹈，非阻塞 |
| `stop_dance` | `dummy`（必填 true） | 清空动作队列 |
| `play_emotion` | `emotion`（42 种意图，省略=random） | 播放预录情绪动作（happy/sad/angry/yes/no/goodbye…） |
| `stop_emotion` | `dummy`（必填 true） | 停止情绪动作 |
| `move_head` | `direction`（left/right/up/down/front） | 头部转向 |
| `sweep_look` | 无 | 头+身体左右环视一周，约 14 秒 |
| `head_tracking` | `enabled`（必填） | 开关人脸跟随 |
| `remember` | `fact`（必填，一句以内） | 保存用户事实到长期记忆（§10） |
| `forget` | `query`（必填，子串匹配） | 删除一条记忆，多条时报告其他候选 |
| `idle_do_nothing` | `reason`（可选） | 空闲轮次保持静止；结果不回传模型 |
| `go_to_sleep` | 无 | 机器人入睡并停止应用（仅在用户明确要求时） |
| `ask_assistant` | `query`（必填） | 把复杂任务委托给家庭助手 OpenClaw（实时信息查询、日程提醒、长期家庭记忆、多步规划）。最近对话（含上次结果）由系统自动注入（5 轮、每轮 200 字），模型只需传 query——history 不进工具 schema，避免 Qwen 截断长函数参数。回复经 markdown/emoji 清洗后回传模型转述；调用期间进入静默等待态（见 §2）。危险指令（删除文件/卸载/发消息/花钱）在工具层直接拦截拒答，不发请求；超时或网络错误返回 `{"ok": false, "error": ...}`，由模型播报兜底话术。需配置 `OPENCLAW_API_URL` + `OPENCLAW_API_TOKEN`，未配置时该工具对模型隐藏 |

### 系统工具（始终启用，独占后台任务管理器引用）

| 工具 | 参数 | 作用 |
|---|---|---|
| `task_status` | `tool_id`（可省略=列出全部） | 查询后台工具任务状态/进度/结果 |
| `task_cancel` | `tool_id`（必填） | 取消一个运行中的后台任务 |

### 预装远程工具（Tool Space，经 MCP 调用，见 §8）

| 工具 | 参数 | 作用 |
|---|---|---|
| `..._search_tool__search_web` | `query`（必填）、`max_results`（1–10，默认 5） | 网络搜索（标题+摘要+链接） |
| `..._time_tool__get_time` | `timezone`、`compare_timezone`（均可空=本地时间） | 查询时区当前时间 |
| `..._weather_tool__get_weather` | `location`（必填） | **仅限今天**的天气；其他日期引导模型改用搜索工具 |

### 外部工具

`REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`（默认 `external_content/external_tools`）下的模块可被发现：profile 引用其名字，或 `AUTOLOAD_EXTERNAL_TOOLS=1` 全量自动加载；与内置工具重名会中止启动。示例 `starter_custom_tool.py` 注意：它 import 的是上游包名 `reachy_mini_conversation_app`，在本 fork 中仅在上游包同时安装时可用。

## 7. 后台任务管理

代码：`tools/background_tool_manager.py`、`tools/tool_constants.py`

- 所有工具调用经此管理器在 asyncio 任务中执行，对话永不因工具阻塞。
- 任务 ID 格式 `{tool_name}-{call_id}-{started_at}`，供 `task_status` / `task_cancel` 使用；状态机 RUNNING / COMPLETED / FAILED / CANCELLED。
- 运行超 24 小时的任务自动取消；已完成任务保留 1 小时后清理；每 300 秒巡检一次。
- 只有系统工具能拿到管理器引用（安全约束：普通工具不能操纵其他任务）。

## 8. Tool Spaces（远程 MCP 工具）

代码：`tool_spaces.py`、`tool_space_routes.py`、`mcp_client.py`

- 把暴露标准 MCP 端点（`https://{owner}-{space}.hf.space/gradio_api/mcp/`）的 **Gradio Space** 安装为远程工具源：工具以 `{别名}__{工具名}` 命名进入本地注册表，调用走网络 MCP，**不在本地下载/执行任何第三方代码**。
- 内置预装三个 Pollen Space（§6 远程工具表），manifest `installed_tool_spaces.json`（v2）已随包附带，启动无需网络发现。
- 安装校验：slug 格式、必须是 HTTPS `.hf.space` 的 `/gradio_api/mcp/` 路径、Space 为 gradio SDK；私有 Space 用 `HF_TOKEN` 鉴权。
- 管理：CLI `tool-spaces add/remove/list`，或 Web UI / JSON-RPC `tool_spaces.add/remove/list`；卸载时自动从所有 profile 禁用对应前缀工具。
- MCP 会话按调用建立/销毁（无常驻连接），调用超时 30 秒，普通调用失败重试一次，超时不重试。

## 9. 人格（Profile）系统

代码：`personality.py`、`profile_store.py`、`profile_toolsets.py`、`personality_routes.py`、`avatars.py`、`prompts.py`

- **来源**：`profiles/` 内置 13 个可见人格（default、bored_teenager、captain_circuit、chess_coach、cosmic_kitchen、hype_bot、mad_scientist_assistant、mars_rover、nature_documentarian、noir_detective、sorry_bro、time_traveler、victorian_butler；`tedai` 为 hidden）+ 外部目录 + 用户自建（`user_personalities/` 前缀，Web 端可增删改）。
- **profile.md 格式**：`+++` 包裹的 TOML frontmatter（`schema_version=1` 必填、`default_tools` 列表、可选 `voice` / `greeting` / `hidden`，未知字段拒绝），正文为人设指令（非空）。旧版 `instructions.txt` 等副文件启动时自动迁移。
- **每人格工具集**：生效工具 = 实例覆盖（`profile_toolsets.json`，Web "Tool access" 页写入）优先于 frontmatter `default_tools`。
- **激活**：切换人格 = 设置 profile + 重建工具注册表 + 热重启会话；"设为默认"持久化到 `startup_settings.json`（`{profile, voice}`），下次启动生效。
- **锁定**：`config.LOCKED_PROFILE`（当前为 `None`）可把整个应用钉死在单一人格，禁用一切切换/编辑。
- **头像**：profile 目录自带 `avatar.svg` → 内置映射（13 个 SVG）→ `default.svg` 兜底。
- **系统提示词组装**（`prompts.py`）：`记忆块 + 人设指令`（记忆在前）；语音取 profile `voice` 否则后端默认。

## 10. 长期记忆

代码：`memory.py`、`tools/remember.py`、`tools/forget.py`

- 存储为 `memory.v1.json`（实例路径或 `$XDG_DATA_HOME/my_conversation_app/`），最多 60 条、每条 280 字符，原子写入、新条目在前；键名 camelCase（与手机端 App 共享的格式）。
- 由 `remember` / `forget` 工具读写；每条带 id、创建时间；forget 按大小写不敏感子串匹配。
- 每次会话（重建）时全部事实以条目列表注入系统提示词开头，指示模型自然利用而非背诵。

## 11. Web UI 与控制接口

代码：`static/`（无框架 ES-module SPA）、`console.py`、`personality_routes.py`、`profile_tool_routes.py`、`tool_space_routes.py`

**HTTP**：`GET /`（SPA）、`/static`（静态资源）、`GET /favicon.ico`；端口 7860。管理面走 **JSON-RPC 2.0 over WebSocket `/rpc`**（无 REST、无 SSE）。

**四个视图**

| 路由 | 视图 | 能做什么 |
|---|---|---|
| `#/` | Talk | 对话光球（点击=静音/取消静音）、实时字幕、回合状态、切人格后的应用中提示 |
| `#/personalities` | Home | 人格卡片网格：切换、新建/编辑/删除自建人格（名称+指令+问候语）、管理工具入口、设为默认 |
| `#/settings` | Settings | HF 连接模式（deployed/local+主机端口）、语音选择（即时生效并持久化）、当前连接状态面板 |
| `#/tools` | Tools | 按 profile 勾选启用工具（内置/外部/Tool Space 分组、恢复默认）、安装/卸载 Tool Space |

**JSON-RPC 方法**

| 组 | 方法 |
|---|---|
| 会话 | `conversation.status` / `conversation.say {text}` / `conversation.interrupt` / `conversation.mic {muted?}` |
| 后端 | `backend.config {hf_mode, hf_host?, hf_port?}` |
| 人格 | `personalities.list / all / load / avatar / save / delete / apply` |
| 语音 | `voices.list / current / apply` |
| 工具集 | `profile_tools.get / save / reset` |
| Tool Spaces | `tool_spaces.list / add / remove` |

**服务端通知**（服务器 → 客户端推送）：`conversation.transcript {role, text, final}`、`conversation.turn {state}`、`conversation.phase {phase, reason}`、`conversation.activity {reason}`、`conversation.level {role, rms}`（约 15 Hz 音量表，驱动光球）。

## 12. 配置参考

环境变量（完整清单，默认值见 `.env.example`）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `REALTIME_BACKEND` | `huggingface` | `dashscope` 切换到 Qwen-Omni 中文后端 |
| `DASHSCOPE_API_KEY` | — | DashScope 必填 |
| `DASHSCOPE_REALTIME_MODEL` | `qwen3.5-omni-flash-realtime` | |
| `DASHSCOPE_REALTIME_WS_BASE` | `wss://dashscope.aliyuncs.com/api-ws/v1` | |
| `DASHSCOPE_REALTIME_VOICE` | `Tina` | 默认音色 |
| `DASHSCOPE_TEMPERATURE` | — | DashScope 会话温度（0-2）。调低可显著提高 flash 模型的工具调用稳定性（视觉提问必调 `camera`）；实测 0.3 表现良好。仅注入 DashScope 会话，HF 后端不受影响 |
| `HF_REALTIME_CONNECTION_MODE` | `deployed` | `deployed` / `local` |
| `HF_REALTIME_WS_URL` | — | local 模式直连地址（base 或完整 realtime URL） |
| `REALTIME_TRANSCRIPTION_LANGUAGE` | `en` | HF 后端输入转写语言（如 `zh`） |
| `HF_TOKEN` | — | HF 鉴权；缺省回退 `hf auth login` |
| `REACHY_MINI_CUSTOM_PROFILE` | — | 启动人选（UI 保存的 `startup_settings.json` 优先；无该文件时 env 生效） |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | `external_content/external_profiles` | 外部人格目录 |
| `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` | `external_content/external_tools` | 外部工具目录 |
| `AUTOLOAD_EXTERNAL_TOOLS` | 关 | 自动加载外部工具目录全部模块 |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | `1440` | 应用级无活动休眠分钟数，0 禁用 |
| `REACHY_MINI_WAKE_WORD_ENABLED` | `1` | 唤醒词门控开关，0 = 常开聆听 |
| `REACHY_MINI_WAKE_WORD_MODELS` | 内置 `hi_reachy.onnx` | 逗号分隔：预训模型名或 `.onnx`/`.tflite` 路径 |
| `REACHY_MINI_WAKE_WORD_THRESHOLD` | `0.5` | 检测阈值 0–1，越低越灵敏 |
| `REACHY_MINI_WAKE_WORD_ACTIVE_TIMEOUT_S` | `300` | 活动空闲退出秒数，0 禁用（ask_assistant 等待期间自动挂起） |
| `REACHY_MINI_GOODBYE_KEYWORDS` | `再见,拜拜,goodbye,bye-bye,bye bye` | 触发 standby 的告别词 |
| `REACHY_MINI_WAKE_WORD_DUMP_DIR` | — | 调试：转储待机麦克风音频为 wav |
| `OPENCLAW_API_URL` | — | OpenClaw 网关的 OpenAI 兼容 completions 地址；未配置则隐藏 `ask_assistant` |
| `OPENCLAW_API_TOKEN` | — | OpenClaw 网关 Bearer token |
| `OPENCLAW_TIMEOUT_S` | `60` | ask_assistant HTTP 超时秒数 |

## 13. 本文档的维护规则

**规则：任何 PR 若新增、修改或删除用户可见行为，必须在同一 PR 中更新本文档对应章节，并刷新顶部"最后核对"的日期与 commit。**

至少涵盖以下变化：

- 工具（新增/改名/参数/行为变化）、后台任务策略
- CLI 参数、环境变量（含默认值变化，同时更新 `.env.example`）
- JSON-RPC 方法与通知、HTTP 端点、Web 视图能力
- 实时后端行为、唤醒词/空闲/休眠策略、动作系统
- profile 格式、人格来源、记忆格式与上限
- Tool Space 校验/安装行为

维护指引：每个章节标注了对应的源文件；编辑时以代码为准，不要照抄旧文档。本文档用中文维护（所有者选择），代码标识符保留英文。
