// Adds a tab bar at the top of the right panel on desktop so the slice strip
// (which used to live in a permanent bottom band) becomes a tab alongside the
// existing controls. The bottom #slices-wrap is hidden on desktop and its
// content is relocated into a SLICES tab that's visible only when active.
//
// Skipped on mobile — mobile uses mobile-drawer.js instead.

const TABS = [
  { id: "panel",   label: "PANEL"  },
  { id: "slices",  label: "SLICES" },
];

export function setupDesktopTabs() {
  // Skip on mobile; mobile drawer handles tabbing.
  if (matchMedia("(max-width: 1023px)").matches) return;

  const rightPanels = document.getElementById("right-panels");
  const slicesWrap  = document.getElementById("slices-wrap");
  if (!rightPanels) return;

  // Build tab bar
  const tabBar = document.createElement("div");
  tabBar.className = "desktop-tabs";
  const tabBtns = TABS.map((t, i) => {
    const b = document.createElement("button");
    b.className = "desktop-tab" + (i === 0 ? " desktop-tab--active" : "");
    b.textContent = t.label;
    b.dataset.id = t.id;
    tabBar.appendChild(b);
    return b;
  });
  rightPanels.insertBefore(tabBar, rightPanels.firstChild);

  // Build SLICES pane and move bottom slices into it
  const slicesPane = document.createElement("div");
  slicesPane.className = "desktop-slices-pane";
  if (slicesWrap) {
    // slicesWrap contains #slices; move the wrapper itself into the pane and
    // override its layout-killing properties so it scrolls inside the panel.
    slicesPane.appendChild(slicesWrap);
    slicesWrap.style.height = "auto";
    slicesWrap.style.borderTop = "0";
  }
  rightPanels.appendChild(slicesPane);

  // Track ALL existing right-panel children so we can show/hide as a group.
  const panelChildren = Array.from(rightPanels.children).filter(
    (el) => el !== tabBar && el !== slicesPane
  );

  function activate(idx) {
    tabBtns.forEach((b, i) => {
      b.classList.toggle("desktop-tab--active", i === idx);
    });
    const showSlices = idx === 1;
    panelChildren.forEach((el) => { el.style.display = showSlices ? "none" : ""; });
    slicesPane.style.display = showSlices ? "flex" : "none";
  }

  tabBtns.forEach((b, i) => b.addEventListener("click", () => activate(i)));
  activate(0);
}
