const BACKEND_LABELS = {
  huggingface: "Hugging Face realtime",
  dashscope: "Alibaba DashScope (Qwen-Omni-Realtime)",
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fetchStatusOnce() {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    let settled = false;
    const socket = new WebSocket(`${protocol}//${location.host}/rpc`);
    const cleanup = () => {
      if (!settled) {
        settled = true;
        socket.close();
      }
    };
    socket.onopen = () => {
      socket.send(
        JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "conversation.status",
          params: {},
        })
      );
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id !== 1) return;
      settled = true;
      socket.close();
      if (message.error) {
        reject(new Error(message.error.message || "status_failed"));
      } else {
        resolve(message.result);
      }
    };
    socket.onerror = () => {
      cleanup();
      reject(new Error("cannot reach /rpc"));
    };
  });
}

async function waitForStatus(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  const deadline = Date.now() + timeoutMs;
  let attempts = 0;
  while (true) {
    attempts += 1;
    try {
      return await fetchStatusOnce();
    } catch (e) {}
    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

function backendLabel(backend) {
  return BACKEND_LABELS[backend] || backend;
}

async function init() {
  const loading = document.getElementById("loading");
  const readyPanel = document.getElementById("configured");
  const guidancePanel = document.getElementById("form-panel");
  const chip = document.getElementById("backend-chip");
  const summary = document.getElementById("backend-summary");
  const guidance = document.getElementById("backend-guidance");

  show(loading, true);
  show(readyPanel, false);
  show(guidancePanel, false);

  const status = await waitForStatus();

  if (status && status.has_key) {
    chip.textContent = backendLabel(status.backend);
    summary.textContent = `The conversation app is running on ${backendLabel(status.backend)}. Speak to the robot to start a conversation.`;
    show(readyPanel, true);
  } else {
    const backend = (status && status.backend) || "huggingface";
    guidance.textContent =
      backend === "dashscope"
        ? "Set DASHSCOPE_API_KEY in the app's .env file, then restart the app."
        : "The Hugging Face backend target is unavailable. Check HF_REALTIME_CONNECTION_MODE and HF_REALTIME_WS_URL in the app's .env file, then restart the app.";
    show(guidancePanel, true);
  }

  show(loading, false);
}

init();
