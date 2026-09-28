/** Faces view: manage the people Reachy recognizes by face. */

import { describeError, inspectFaceCamera, labelFace, listFaces, removeFace, renameFace, setFaceNicknames } from "../api.js";
import { h } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";

const CAMERA_CHECK_TARGET_MS = 120; // ~8 fps; the round trip runs first, only the remainder waits
const CAMERA_CHECK_MAX_CONSECUTIVE_ERRORS = 3; // a transient blip (handler rebuild) must not kill the panel

function formatDate(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;
  return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function drawFaceScores(ctx, face, { threshold, progressiveThreshold }) {
  const bbox = (face.bbox || []).map(Number);
  if (bbox.length !== 4 || !bbox.every(Number.isFinite)) return;
  const [x, y, w, boxHeight] = bbox;
  const scores = Array.isArray(face.scores) ? face.scores : [];
  const best = scores[0] || {};
  const bestScore = Number(best.similarity) || 0;
  const matched = bestScore >= threshold;
  // Amber marks the zone where the conversation loop's own recognize() would
  // save this frame into the matched person's reference set.
  const color = bestScore >= progressiveThreshold ? "#f59e0b" : matched ? "#34d399" : "#848b98";

  // Scale the overlay with the frame: a fixed px size reads fine at 640-wide
  // but vanishes on higher-resolution cameras.
  const fontSize = Math.max(16, Math.round(ctx.canvas.width / 32));
  const lineHeight = Math.round(fontSize * 1.3);
  const padding = Math.round(fontSize * 0.3);

  ctx.lineWidth = Math.max(2, fontSize / 8);
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, boxHeight);

  const lines = [
    matched ? `${best.name} ${Number(best.similarity).toFixed(3)}` : `unknown (best ${bestScore.toFixed(3)})`,
    ...scores.slice(1).map((entry) => `${entry.name} ${Number(entry.similarity).toFixed(3)}`),
  ];
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  const textWidth = Math.max(...lines.map((line) => ctx.measureText(line).width));
  let textTop = y - lines.length * lineHeight - padding;
  if (textTop < 0) textTop = y + padding;
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(x, textTop, textWidth + 2 * padding, lines.length * lineHeight + padding);
  lines.forEach((line, index) => {
    ctx.fillStyle = index === 0 ? color : "#b8bdc7";
    ctx.fillText(line, x + padding, textTop + (index + 0.75) * lineHeight);
  });
}

function drawInspection(canvas, payload) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);
      const threshold = Number(payload?.threshold) || 0;
      const progressiveThreshold = Number(payload?.progressiveThreshold) || threshold;
      for (const face of payload?.faces || []) {
        drawFaceScores(ctx, face, { threshold, progressiveThreshold });
      }
      resolve();
    };
    image.onerror = () => resolve();
    image.src = payload?.image || "";
  });
}

function mountFaceCheckPanel({ signal, onLabeled, getPeople }) {
  const status = h("span", { class: "settings-hint", role: "status", "aria-live": "polite" });
  const startButton = h("button", { type: "button", class: "btn btn--primary" }, "Start camera check");
  const freezeButton = h("button", { type: "button", class: "btn btn--ghost", disabled: "" }, "Freeze");
  const canvas = h("canvas", {
    class: "face-check-canvas",
    role: "img",
    "aria-label": "Live camera view with each face's match score against every enrolled person",
  });
  const labelList = h("div", { class: "face-check-faces" });
  const stage = h("div", { class: "face-check", hidden: "" }, canvas, labelList);
  const root = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Live camera check"),
    h(
      "p",
      { class: "settings-hint" },
      "Each face Reachy's camera sees, scored against every enrolled person — nothing is saved " +
        "automatically. Freeze a frame to label one of its faces: an existing person gains the " +
        "reference, a new name enrolls them."
    ),
    h("div", { class: "face-check-controls" }, startButton, freezeButton, status),
    stage
  );

  let mode = "idle"; // idle | running | frozen
  let lastPayload = null;
  let frozenPayload = null;

  async function poll() {
    let consecutiveErrors = 0;
    while (mode === "running" && !signal?.aborted) {
      const startedAt = performance.now();
      try {
        const payload = await inspectFaceCamera();
        if (mode !== "running" || signal?.aborted) return;
        consecutiveErrors = 0;
        lastPayload = payload;
        stage.hidden = false;
        status.textContent = "";
        await drawInspection(canvas, payload);
      } catch (error) {
        if (mode !== "running" || signal?.aborted) return;
        consecutiveErrors += 1;
        if (consecutiveErrors >= CAMERA_CHECK_MAX_CONSECUTIVE_ERRORS) {
          stopToIdle(`Camera check failed: ${describeError(error)}`);
          return;
        }
      }
      const elapsed = performance.now() - startedAt;
      await new Promise((resolve) => setTimeout(resolve, Math.max(0, CAMERA_CHECK_TARGET_MS - elapsed)));
    }
  }

  function start() {
    mode = "running";
    frozenPayload = null;
    labelList.replaceChildren();
    startButton.textContent = "Stop";
    freezeButton.disabled = false;
    freezeButton.textContent = "Freeze";
    status.textContent = "Loading camera…";
    poll();
  }

  function stopToIdle(message) {
    mode = "idle";
    frozenPayload = null;
    labelList.replaceChildren();
    startButton.textContent = "Start camera check";
    freezeButton.disabled = true;
    freezeButton.textContent = "Freeze";
    stage.hidden = true;
    if (message && !signal?.aborted) status.textContent = message;
  }

  function freeze() {
    if (mode !== "running" || !lastPayload) return;
    frozenPayload = lastPayload;
    mode = "frozen";
    startButton.textContent = "Stop";
    freezeButton.disabled = false;
    freezeButton.textContent = "Cancel freeze";
    status.textContent = "Frozen. Label a face below, or cancel the freeze.";
    renderLabelRows();
  }

  function startLabelForm(row, faceIndex) {
    const existing = getPeople?.() || [];
    const personOptions = existing.map((person) => {
      const nicknames = Array.isArray(person.nicknames) ? person.nicknames.filter(Boolean) : [];
      const label = nicknames.length ? `${person.name}（${nicknames.join(" / ")}）` : person.name;
      return h("option", { value: person.name }, label);
    });
    const select = h(
      "select",
      { class: "settings-select face-check-label-select", "aria-label": `Existing person for face ${faceIndex + 1}` },
      h("option", { value: "" }, "选择已有的人…"),
      ...personOptions,
      h("option", { value: "__new__" }, "新建一个人…")
    );
    const input = h("input", {
      type: "text",
      class: "settings-input",
      maxlength: "24",
      placeholder: "Name",
      "aria-label": `Name for face ${faceIndex + 1}`,
      autocomplete: "off",
    });
    const cancelButton = h("button", { type: "button", class: "btn btn--ghost" }, "Cancel");
    const form = h(
      "form",
      { class: "face-check-label-form" },
      personOptions.length ? select : null,
      input,
      h("button", { type: "submit", class: "btn btn--primary" }, "Save"),
      cancelButton
    );
    row.replaceChildren(form);

    select.addEventListener("change", () => {
      if (select.value === "__new__") {
        input.value = "";
        input.focus();
      } else if (select.value) {
        input.value = select.value;
      }
    });
    if (!personOptions.length) input.focus();

    cancelButton.addEventListener("click", renderLabelRows);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const name = input.value.trim();
      if (!name) return;
      try {
        const result = await labelFace(frozenPayload?.frameId, faceIndex, name);
        if (signal?.aborted) return;
        status.textContent = `Saved as “${result?.face?.name || name}”.`;
        onLabeled?.();
        renderLabelRows();
      } catch (error) {
        if (signal?.aborted) return;
        status.textContent = `Failed to label: ${describeError(error)}`;
      }
    });
  }

  function renderLabelRows() {
    const faces = frozenPayload?.faces || [];
    labelList.replaceChildren();
    if (!faces.length) {
      labelList.append(h("p", { class: "settings-hint" }, "No faces in this frame."));
      return;
    }
    faces.forEach((face, index) => {
      const best = (face.scores || [])[0];
      const summary = best ? `${best.name} · ${Number(best.similarity).toFixed(3)}` : "no scores";
      const labelButton = h("button", { type: "button", class: "btn btn--ghost" }, "Label as…");
      const row = h(
        "div",
        { class: "face-check-face-row" },
        h("span", { class: "face-check-face-index" }, `#${index + 1}`),
        h("span", { class: "face-check-face-summary" }, summary),
        labelButton
      );
      labelButton.addEventListener("click", () => startLabelForm(row, index));
      labelList.append(row);
    });
  }

  startButton.addEventListener("click", () => {
    if (mode === "running" || mode === "frozen") stopToIdle("");
    else start();
  });
  freezeButton.addEventListener("click", () => {
    if (mode === "frozen") start();
    else freeze();
  });
  signal?.addEventListener("abort", () => {
    mode = "idle";
  });
  return root;
}

function describeFace(face) {
  const details = [];
  const nicknames = Array.isArray(face.nicknames) ? face.nicknames.filter(Boolean) : [];
  if (nicknames.length) details.push(`Also called ${nicknames.join(" · ")}`);
  details.push(`${Number(face.embeddingCount) || 0} reference photos`);
  const lastSeen = formatDate(face.lastSeenAt);
  if (lastSeen) details.push(`Last seen ${lastSeen}`);
  return details.join(" · ");
}

export async function mountFacesView({ outlet, signal }) {
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const list = h(
    "div",
    { class: "settings-faces", role: "list", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading enrolled faces…")
  );
  // The camera panel's label form offers these as picks; render() keeps it current.
  let people = [];
  const view = h(
    "section",
    { class: "view view--faces" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "People"),
      h(
        "p",
        { class: "view-subtitle" },
        "Faces Reachy recognizes. Enroll someone by saying 我叫XX，记住我 to the robot."
      )
    ),
    h(
      "section",
      { class: "settings-section" },
      h("h2", { class: "settings-section-title" }, "Recognized people"),
      list,
      status
    ),
    mountFaceCheckPanel({ signal, onLabeled: refresh, getPeople: () => people })
  );
  outlet.replaceChildren(view);

  let busy = false;

  function setBusy(nextBusy) {
    busy = nextBusy;
    list.toggleAttribute("aria-busy", nextBusy);
    list.querySelectorAll("button").forEach((button) => {
      button.disabled = nextBusy;
    });
  }

  function render(faces) {
    people = Array.isArray(faces) ? faces : [];
    list.replaceChildren();
    if (!faces.length) {
      list.appendChild(
        h("p", { class: "settings-hint" }, "No faces enrolled yet. Talk to Reachy to add someone.")
      );
      return;
    }
    for (const face of faces) {
      const renameButton = h("button", { type: "button", class: "btn btn--ghost" }, "Rename");
      const nicknamesButton = h("button", { type: "button", class: "btn btn--ghost" }, "Nicknames");
      const removeButton = h("button", { type: "button", class: "btn btn--ghost" }, "Remove");
      const row = h(
        "div",
        { class: "settings-tool-space", role: "listitem" },
        h(
          "div",
          { class: "settings-tool-space-summary" },
          h("strong", { class: "settings-tool-space-name" }, face.name),
          h("span", { class: "settings-tool-space-meta" }, describeFace(face))
        ),
        h("div", { class: "settings-tool-space-controls" }, renameButton, nicknamesButton, removeButton)
      );

      renameButton.addEventListener("click", () => {
        if (!busy) startRename(row, face);
      });

      nicknamesButton.addEventListener("click", () => {
        if (!busy) startNicknames(row, face);
      });

      removeButton.addEventListener("click", async () => {
        if (busy) return;
        const confirmed = await confirmDialog({
          title: "Remove face?",
          message: `Reachy will no longer recognize “${face.name}”.`,
          confirmLabel: "Remove",
          danger: true,
          signal,
        });
        if (!confirmed || signal?.aborted) return;

        status.classList.remove("is-error");
        status.textContent = `Removing “${face.name}”…`;
        setBusy(true);
        try {
          const result = await removeFace(face.id);
          if (signal?.aborted) return;
          status.textContent = result?.ok ? `Removed “${face.name}”.` : "Removed.";
          await refresh();
        } catch (error) {
          if (signal?.aborted) return;
          status.textContent = `Failed to remove: ${describeError(error)}`;
          status.classList.add("is-error");
        } finally {
          if (!signal?.aborted) setBusy(false);
        }
      });

      list.appendChild(row);
    }
  }

  function startRename(row, face) {
    const input = h("input", {
      type: "text",
      class: "settings-input",
      value: face.name,
      maxlength: "24",
      "aria-label": `New name for ${face.name}`,
      autocomplete: "off",
    });
    const cancelButton = h("button", { type: "button", class: "btn btn--ghost" }, "Cancel");
    const form = h(
      "form",
      { class: "settings-tool-space-controls" },
      input,
      h("button", { type: "submit", class: "btn btn--primary" }, "Save"),
      cancelButton
    );
    row.replaceChild(form, row.lastChild);
    input.focus();
    input.select();

    cancelButton.addEventListener("click", () => refresh());

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const name = input.value.trim();
      if (!name || name === face.name) {
        refresh();
        return;
      }
      status.classList.remove("is-error");
      setBusy(true);
      try {
        await renameFace(face.id, name);
        if (signal?.aborted) return;
        status.textContent = `Renamed to “${name}”.`;
        await refresh();
      } catch (error) {
        if (signal?.aborted) return;
        status.textContent = `Failed to rename: ${describeError(error)}`;
        status.classList.add("is-error");
        refresh();
      } finally {
        if (!signal?.aborted) setBusy(false);
      }
    });
  }

  function startNicknames(row, face) {
    const current = Array.isArray(face.nicknames) ? face.nicknames.filter(Boolean) : [];
    const input = h("input", {
      type: "text",
      class: "settings-input",
      value: current.join(", "),
      maxlength: "110",
      placeholder: "e.g. 老凯, Kai",
      "aria-label": `Extra names to call ${face.name}, comma-separated`,
      autocomplete: "off",
    });
    const cancelButton = h("button", { type: "button", class: "btn btn--ghost" }, "Cancel");
    const form = h(
      "form",
      { class: "settings-tool-space-controls" },
      input,
      h("button", { type: "submit", class: "btn btn--primary" }, "Save"),
      cancelButton
    );
    row.replaceChild(form, row.lastChild);
    input.focus();

    cancelButton.addEventListener("click", () => refresh());

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const nicknames = input.value
        .split(/[,，、]/)
        .map((part) => part.trim())
        .filter(Boolean);
      const unchanged =
        nicknames.length === current.length && nicknames.every((name, index) => name === current[index]);
      if (unchanged) {
        refresh();
        return;
      }
      status.classList.remove("is-error");
      setBusy(true);
      try {
        const result = await setFaceNicknames(face.id, nicknames);
        if (signal?.aborted) return;
        // The server dedupes and caps; report what was actually saved.
        const saved = Array.isArray(result?.face?.nicknames) ? result.face.nicknames : nicknames;
        status.textContent = saved.length
          ? `Reachy now calls ${face.name} by ${[face.name, ...saved].join(", ")} at random.`
          : `Cleared the extra names for ${face.name}.`;
        await refresh();
      } catch (error) {
        if (signal?.aborted) return;
        status.textContent = `Failed to save nicknames: ${describeError(error)}`;
        status.classList.add("is-error");
        refresh();
      } finally {
        if (!signal?.aborted) setBusy(false);
      }
    });
  }

  async function refresh() {
    try {
      const payload = await listFaces();
      if (signal?.aborted) return;
      render(Array.isArray(payload?.faces) ? payload.faces : []);
    } catch (error) {
      if (signal?.aborted) return;
      list.replaceChildren(
        h("p", { class: "settings-status is-error" }, `Could not load faces: ${describeError(error)}`)
      );
    }
  }

  await refresh();
}
