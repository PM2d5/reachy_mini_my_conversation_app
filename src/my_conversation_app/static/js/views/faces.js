/** Faces view: manage the people Reachy recognizes by face. */

import { describeError, listFaces, removeFace, renameFace } from "../api.js";
import { h } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";

function formatDate(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value <= 0) return null;
  return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function describeFace(face) {
  const details = [`${Number(face.embeddingCount) || 0} reference photos`];
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
    )
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
    list.replaceChildren();
    if (!faces.length) {
      list.appendChild(
        h("p", { class: "settings-hint" }, "No faces enrolled yet. Talk to Reachy to add someone.")
      );
      return;
    }
    for (const face of faces) {
      const renameButton = h("button", { type: "button", class: "btn btn--ghost" }, "Rename");
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
        h("div", { class: "settings-tool-space-controls" }, renameButton, removeButton)
      );

      renameButton.addEventListener("click", () => {
        if (!busy) startRename(row, face);
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
