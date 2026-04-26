// History timeline + scrubber UI
// Exports: setupHistoryUI(rootEl, store, ledMeshes)
// store: createHistoryStore() instance from history-store.js
// ledMeshes: Map<"chain:pos", THREE.Mesh> from main.js (window.twin.ledMeshes)

const ACCENT_RED = "#ff4444";
const BG_PANEL   = "#0d1117";
const BG_SECTION = "#111820";
const BORDER     = "#1e2d3d";
const TEXT       = "#c9d8e8";
const TEXT_DIM   = "#546e7a";

const POLL_INTERVAL_MS = 5000;

export function setupHistoryUI(rootEl, store, ledMeshes) {
  rootEl.style.cssText = [
    "display:flex",
    "width:100%",
    "height:100%",
    `background:${BG_PANEL}`,
    `border-top:1px solid ${BORDER}`,
    "font-family:ui-monospace,'Cascadia Code','Fira Mono',monospace",
    "font-size:11px",
    `color:${TEXT}`,
    "overflow:hidden",
  ].join(";");

  // ── Left: inference list ────────────────────────────────────────────────────
  const listPane = el("div", {
    width: "220px", minWidth: "180px", flexShrink: "0",
    display: "flex", flexDirection: "column",
    borderRight: `1px solid ${BORDER}`, overflow: "hidden",
  });

  const listHeader = el("div", {
    padding: "6px 10px", fontSize: "9px", letterSpacing: "2px",
    color: TEXT_DIM, borderBottom: `1px solid ${BORDER}`,
    background: BG_SECTION, flexShrink: "0",
  });
  listHeader.textContent = "HISTORY";

  const listScroll = el("div", {
    flex: "1", overflowY: "auto",
    scrollbarWidth: "thin", scrollbarColor: `${BORDER} transparent`,
  });

  listPane.appendChild(listHeader);
  listPane.appendChild(listScroll);

  // ── Right: scrubber + controls ──────────────────────────────────────────────
  const scrubPane = el("div", {
    flex: "1", display: "flex", flexDirection: "column",
    padding: "10px 14px", gap: "8px", overflow: "hidden",
  });

  const scrubHeader = el("div", {
    fontSize: "9px", letterSpacing: "2px", color: TEXT_DIM, flexShrink: "0",
  });
  scrubHeader.textContent = "SCRUBBER";

  const frameLabel = el("div", {
    fontSize: "10px", color: TEXT_DIM,
    fontVariantNumeric: "tabular-nums", flexShrink: "0",
  });
  frameLabel.textContent = "No record selected";

  const sliderRow = el("div", {
    display: "flex", alignItems: "center", gap: "8px", flexShrink: "0",
  });

  const scrubSlider = document.createElement("input");
  scrubSlider.type = "range";
  scrubSlider.min = "0";
  scrubSlider.max = "0";
  scrubSlider.value = "0";
  scrubSlider.disabled = true;
  Object.assign(scrubSlider.style, {
    flex: "1", accentColor: ACCENT_RED, height: "4px", cursor: "pointer",
  });

  const scrubIndex = el("span", {
    minWidth: "50px", textAlign: "right", color: ACCENT_RED,
    fontVariantNumeric: "tabular-nums", fontSize: "10px",
  });
  scrubIndex.textContent = "0 / 0";

  sliderRow.appendChild(scrubSlider);
  sliderRow.appendChild(scrubIndex);

  const btnRow = el("div", { display: "flex", gap: "6px", flexShrink: "0" });

  const btnPlay = makeBtn("PLAY",    ACCENT_RED, "#080c10");
  const btnLive = makeBtn("LIVE",    "#00c8ff",  "#080c10");
  const btnDiff = makeBtn("DIFF OFF", BG_SECTION, "#00c8ff");
  btnDiff.style.border = "1px solid #005b74";

  btnRow.appendChild(btnPlay);
  btnRow.appendChild(btnLive);
  btnRow.appendChild(btnDiff);

  scrubPane.appendChild(scrubHeader);
  scrubPane.appendChild(frameLabel);
  scrubPane.appendChild(sliderRow);
  scrubPane.appendChild(btnRow);

  rootEl.appendChild(listPane);
  rootEl.appendChild(scrubPane);

  // ── State ────────────────────────────────────────────────────────────────────
  let selectedRecord = null;
  let playing        = false;
  let liveMode       = true;
  let diffMode       = false;
  let faultState     = null;
  let playTimer      = null;
  let pollTimer      = null;

  // ── Row builder ──────────────────────────────────────────────────────────────

  function makeRowEl(rec) {
    const row = el("div", {
      display: "flex", alignItems: "center", gap: "8px",
      padding: "6px 10px", cursor: "pointer",
      borderLeft: "2px solid transparent",
      borderBottom: `1px solid ${BORDER}`,
      transition: "background 0.1s",
    });
    row.dataset.id = rec.id;

    const digit = el("span", {
      fontSize: "28px", fontWeight: "700", color: ACCENT_RED,
      minWidth: "28px", textAlign: "center", lineHeight: "1",
    });
    digit.textContent = rec.predicted_digit != null ? String(rec.predicted_digit) : "?";

    const meta = el("div", {
      display: "flex", flexDirection: "column", gap: "2px",
      flex: "1", overflow: "hidden",
    });

    const ts = el("span", {
      fontSize: "9px", color: TEXT_DIM, letterSpacing: "0.3px",
      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      fontVariantNumeric: "tabular-nums",
    });
    ts.textContent = formatTs(rec.timestamp);

    const pill = el("span", {
      display: "inline-block", background: BG_SECTION,
      border: `1px solid ${BORDER}`, borderRadius: "10px",
      padding: "1px 6px", fontSize: "8px", color: TEXT_DIM,
      letterSpacing: "0.5px", width: "fit-content",
    });
    pill.textContent = `${rec.n_frames ?? 0} frames`;

    meta.appendChild(ts);
    meta.appendChild(pill);
    row.appendChild(digit);
    row.appendChild(meta);

    row.addEventListener("mouseenter", () => {
      if (row.dataset.id !== String(selectedRecord?.id)) {
        row.style.background = BG_SECTION;
      }
    });
    row.addEventListener("mouseleave", () => {
      if (row.dataset.id !== String(selectedRecord?.id)) {
        row.style.background = "";
      }
    });
    row.addEventListener("click", () => selectRecord(rec));
    return row;
  }

  function formatTs(ts) {
    if (!ts) return "--";
    try {
      return new Date(ts).toISOString().replace("T", " ").slice(0, 19);
    } catch { return String(ts); }
  }

  // ── Record selection ─────────────────────────────────────────────────────────

  async function selectRecord(rec) {
    let fullRec = rec;
    if (store.getRecord) {
      try { fullRec = (await store.getRecord(rec.id)) ?? rec; } catch { /* use rec */ }
    }
    selectedRecord = fullRec;
    liveMode = false;
    stopPolling();
    stopPlayback();

    const nFrames = fullRec.n_frames ?? fullRec.frames?.length ?? 0;
    scrubSlider.disabled = nFrames === 0;
    scrubSlider.max   = String(Math.max(0, nFrames - 1));
    scrubSlider.value = "0";
    scrubIndex.textContent = `0 / ${nFrames}`;
    frameLabel.textContent = `Digit: ${fullRec.predicted_digit ?? "?"} — ${nFrames} frames`;

    highlightRow(rec.id);
    applyFrames(fullRec, 0);

    if (diffMode) {
      await fetchFaultState();
      drawDiffOverlay(fullRec, 0);
    }
  }

  function highlightRow(id) {
    for (const rowEl of listScroll.querySelectorAll("[data-id]")) {
      const sel = rowEl.dataset.id === String(id);
      rowEl.style.background  = sel ? "#181f2a" : "";
      rowEl.style.borderLeft  = sel ? `2px solid ${ACCENT_RED}` : "2px solid transparent";
    }
  }

  // ── Frame replay ─────────────────────────────────────────────────────────────

  function applyFrames(rec, frameIdx) {
    for (const mesh of ledMeshes.values()) {
      mesh.material.color.setRGB(0.02, 0.03, 0.063);
    }
    const frames = rec.frames ?? [];
    const limit  = Math.min(frameIdx, frames.length - 1);
    for (let i = 0; i <= limit; i++) {
      const frame = frames[i];
      if (!frame?.deltas) continue;
      for (const [chain, position, r, g, b] of frame.deltas) {
        const mesh = ledMeshes.get(`${chain}:${position}`);
        if (mesh) mesh.material.color.setRGB(r / 255, g / 255, b / 255);
      }
    }
  }

  // ── Diff overlay ─────────────────────────────────────────────────────────────

  async function fetchFaultState() {
    try {
      const resp = await fetch("/fault/state");
      if (resp.ok) faultState = await resp.json();
    } catch { faultState = null; }
  }

  function drawDiffOverlay(rec, frameIdx) {
    if (!faultState) return;
    const commanded = new Map();
    const frames = rec.frames ?? [];
    const limit  = Math.min(frameIdx, frames.length - 1);
    for (let i = 0; i <= limit; i++) {
      const frame = frames[i];
      if (!frame?.deltas) continue;
      for (const [chain, position, r, g, b] of frame.deltas) {
        commanded.set(`${chain}:${position}`, [r, g, b]);
      }
    }

    const faultyKeys = new Set(
      (faultState.faults ?? []).map(f => `${f.chain}:${f.position}`)
    );

    for (const [key, [cr, cg, cb]] of commanded.entries()) {
      const mesh = ledMeshes.get(key);
      if (!mesh) continue;
      const mismatch = faultyKeys.has(key) && (cr > 10 || cg > 10 || cb > 10);
      mesh.material.wireframe = mismatch;
      if (mismatch) mesh.material.color.setRGB(1, 0.27, 0.27);
    }
  }

  function clearDiffOverlay() {
    for (const mesh of ledMeshes.values()) {
      mesh.material.wireframe = false;
    }
  }

  // ── Scrubber events ──────────────────────────────────────────────────────────

  scrubSlider.addEventListener("input", () => {
    if (!selectedRecord) return;
    const idx = Number(scrubSlider.value);
    const nFrames = selectedRecord.n_frames ?? selectedRecord.frames?.length ?? 0;
    scrubIndex.textContent = `${idx} / ${nFrames}`;
    applyFrames(selectedRecord, idx);
    if (diffMode) drawDiffOverlay(selectedRecord, idx);
  });

  // ── Play / Pause ─────────────────────────────────────────────────────────────

  function startPlayback() {
    if (!selectedRecord || playing) return;
    playing = true;
    btnPlay.textContent = "PAUSE";
    const nFrames = selectedRecord.n_frames ?? selectedRecord.frames?.length ?? 0;
    if (nFrames === 0) return;

    playTimer = setInterval(() => {
      const cur  = Number(scrubSlider.value);
      const next = cur + 1;
      if (next >= nFrames) { stopPlayback(); return; }
      scrubSlider.value = String(next);
      scrubIndex.textContent = `${next} / ${nFrames}`;
      applyFrames(selectedRecord, next);
      if (diffMode) drawDiffOverlay(selectedRecord, next);
    }, 80);
  }

  function stopPlayback() {
    playing = false;
    btnPlay.textContent = "PLAY";
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
  }

  btnPlay.addEventListener("click", () => {
    if (!selectedRecord) return;
    if (playing) stopPlayback(); else startPlayback();
  });

  // ── Live mode ────────────────────────────────────────────────────────────────

  function enterLiveMode() {
    liveMode = true;
    scrubSlider.disabled = true;
    scrubSlider.value = "0";
    scrubIndex.textContent = "-- / --";
    frameLabel.textContent = "Live mode";
    stopPlayback();
    clearDiffOverlay();
    selectedRecord = null;
    highlightRow(-1);
    startPolling();
  }

  btnLive.addEventListener("click", enterLiveMode);

  // ── Diff toggle ──────────────────────────────────────────────────────────────

  btnDiff.addEventListener("click", async () => {
    diffMode = !diffMode;
    btnDiff.textContent      = diffMode ? "DIFF ON"  : "DIFF OFF";
    btnDiff.style.background = diffMode ? "#1a2820"  : BG_SECTION;
    btnDiff.style.borderColor = diffMode ? "#00e676" : "#005b74";
    btnDiff.style.color      = diffMode ? "#00e676"  : "#00c8ff";

    if (diffMode && selectedRecord) {
      await fetchFaultState();
      drawDiffOverlay(selectedRecord, Number(scrubSlider.value));
    } else {
      clearDiffOverlay();
      if (selectedRecord) applyFrames(selectedRecord, Number(scrubSlider.value));
    }
  });

  // ── List rendering ───────────────────────────────────────────────────────────

  function renderList(records) {
    while (listScroll.firstChild) listScroll.removeChild(listScroll.firstChild);

    if (!records?.length) {
      const empty = el("div", { padding: "12px 10px", color: TEXT_DIM, fontSize: "10px" });
      empty.textContent = "No inferences recorded yet.";
      listScroll.appendChild(empty);
      return;
    }
    for (const rec of records) {
      listScroll.appendChild(makeRowEl(rec));
    }
    if (selectedRecord) highlightRow(selectedRecord.id);
  }

  // ── Polling ──────────────────────────────────────────────────────────────────

  async function refreshList() {
    if (!liveMode) return;
    try {
      const records = await store.listRecords();
      renderList(records);
    } catch { /* ignore */ }
  }

  function startPolling() {
    if (pollTimer) return;
    refreshList();
    pollTimer = setInterval(refreshList, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  // ── Init ─────────────────────────────────────────────────────────────────────

  if (store.onRecord) {
    store.onRecord(() => { if (liveMode) refreshList(); });
  }

  enterLiveMode();

  return { enterLiveMode, stopPolling };
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function el(tag, styles) {
  const node = document.createElement(tag);
  if (styles) Object.assign(node.style, styles);
  return node;
}

function makeBtn(label, bg, color) {
  const btn = el("button", {
    padding: "6px 10px", background: bg, color,
    border: "none", borderRadius: "5px",
    fontFamily: "ui-monospace,'Cascadia Code','Fira Mono',monospace",
    fontSize: "9px", letterSpacing: "1.5px", fontWeight: "600",
    cursor: "pointer", transition: "opacity 0.15s, transform 0.05s",
  });
  btn.textContent = label;
  btn.addEventListener("mouseenter", () => { btn.style.opacity = "0.8"; });
  btn.addEventListener("mouseleave", () => { btn.style.opacity = "1"; });
  btn.addEventListener("mousedown",  () => { btn.style.transform = "scale(0.96)"; });
  btn.addEventListener("mouseup",    () => { btn.style.transform = ""; });
  return btn;
}
