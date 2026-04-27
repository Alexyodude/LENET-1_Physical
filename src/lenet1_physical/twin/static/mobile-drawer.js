// Mobile slide-up drawer with CONTROLS / FAULTS / HISTORY / SLICES tabs.
// Only activates when viewport width <= 768px.
// Called by worker-7 at the end of init(), after all panels exist.

const TABS = ["CONTROLS", "FAULTS", "HISTORY", "SLICES"];
const SWIPE_THRESHOLD = 40; // px

export function setupMobileDrawer() {
  if (!matchMedia("(max-width: 1023px)").matches) return;

  // ── Build DOM ──────────────────────────────────────────────────────────────
  const drawer = document.createElement("div");
  drawer.className = "mobile-drawer";
  drawer.setAttribute("role", "region");
  drawer.setAttribute("aria-label", "Controls drawer");

  // Drag handle pill
  const handleBar = document.createElement("div");
  handleBar.className = "mobile-drawer__handle-bar";
  const pill = document.createElement("div");
  pill.className = "mobile-drawer__pill";
  handleBar.appendChild(pill);
  drawer.appendChild(handleBar);

  // Tab bar
  const tabBar = document.createElement("div");
  tabBar.className = "mobile-drawer__tabs";
  const tabBtns = TABS.map((label, i) => {
    const btn = document.createElement("button");
    btn.className = "mobile-drawer__tab" + (i === 0 ? " mobile-drawer__tab--active" : "");
    btn.textContent = label;
    btn.setAttribute("aria-selected", i === 0 ? "true" : "false");
    tabBar.appendChild(btn);
    return btn;
  });
  drawer.appendChild(tabBar);

  // Body with one pane per tab
  const body = document.createElement("div");
  body.className = "mobile-drawer__body";
  const panes = TABS.map((_, i) => {
    const pane = document.createElement("div");
    pane.className = "mobile-drawer__pane" + (i === 0 ? " mobile-drawer__pane--active" : "");
    body.appendChild(pane);
    return pane;
  });
  drawer.appendChild(body);

  document.body.appendChild(drawer);

  // ── Move existing panel content into panes ─────────────────────────────────
  // Tab 0 — CONTROLS: all .panel-section elements from #panel
  const panel = document.getElementById("panel");
  if (panel) {
    panel.querySelectorAll(".panel-section").forEach(sec => panes[0].appendChild(sec));
  }

  // Tab 1 — FAULTS
  const faultHost = document.getElementById("fault-panel-host");
  if (faultHost) panes[1].appendChild(faultHost);

  // Tab 2 — HISTORY
  const historyHost = document.getElementById("history-panel-host");
  if (historyHost) panes[2].appendChild(historyHost);

  // Tab 3 — SLICES: arch-panel-host + draw-host + #slices-wrap contents
  const archHost = document.getElementById("arch-panel-host");
  if (archHost) panes[3].appendChild(archHost);

  const drawHost = document.getElementById("draw-host");
  if (drawHost) panes[3].appendChild(drawHost);

  const slicesWrap = document.getElementById("slices-wrap");
  if (slicesWrap) panes[3].appendChild(slicesWrap);

  // Hide desktop containers (CSS also does this, belt-and-suspenders)
  const rightPanels = document.getElementById("right-panels");
  if (rightPanels) rightPanels.style.display = "none";
  const resizer = document.getElementById("panel-resizer");
  if (resizer) resizer.style.display = "none";

  // ── Tab switching ──────────────────────────────────────────────────────────
  let activeTab = 0;

  function activateTab(idx) {
    tabBtns[activeTab].classList.remove("mobile-drawer__tab--active");
    tabBtns[activeTab].setAttribute("aria-selected", "false");
    panes[activeTab].classList.remove("mobile-drawer__pane--active");

    activeTab = idx;
    tabBtns[activeTab].classList.add("mobile-drawer__tab--active");
    tabBtns[activeTab].setAttribute("aria-selected", "true");
    panes[activeTab].classList.add("mobile-drawer__pane--active");

    // Expand drawer when a tab is tapped while collapsed
    if (!expanded) setExpanded(true);
  }

  tabBtns.forEach((btn, i) => {
    btn.addEventListener("click", () => activateTab(i));
  });

  // ── Expand / collapse ──────────────────────────────────────────────────────
  let expanded = false;

  function setExpanded(val) {
    expanded = val;
    drawer.classList.toggle("mobile-drawer--expanded", expanded);
    document.body.classList.toggle("drawer-expanded", expanded);
  }

  handleBar.addEventListener("click", () => setExpanded(!expanded));

  // ── Touch / pointer swipe ──────────────────────────────────────────────────
  let pointerStartY = null;
  let pointerStartExpanded = false;

  drawer.addEventListener("pointerdown", (e) => {
    // Only track drags that start on the handle or tab bar (not inside body scroll)
    if (body.contains(e.target) && e.target !== body) return;
    pointerStartY = e.clientY;
    pointerStartExpanded = expanded;
    drawer.setPointerCapture(e.pointerId);
  });

  drawer.addEventListener("pointermove", (e) => {
    if (pointerStartY === null) return;
    const dy = pointerStartY - e.clientY; // positive = swipe up
    if (!pointerStartExpanded && dy > SWIPE_THRESHOLD) setExpanded(true);
    if (pointerStartExpanded && dy < -SWIPE_THRESHOLD) setExpanded(false);
  });

  drawer.addEventListener("pointerup", () => { pointerStartY = null; });
  drawer.addEventListener("pointercancel", () => { pointerStartY = null; });
}
