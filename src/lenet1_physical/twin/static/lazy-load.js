export function isMobile() {
  return window.matchMedia('(max-width: 768px)').matches;
}

export function setupFrameDebounce() {
  const pending = new Map();
  let rafScheduled = false;

  function flush() {
    rafScheduled = false;
    for (const [layer, frame] of pending) {
      pending.delete(layer);
      const evt = new CustomEvent('frame', { detail: frame });
      window.twinEvents.dispatchEvent(evt);
    }
  }

  function dispatchCoalescedFrame(frame) {
    const layer = frame.layer ?? '__default__';
    pending.set(layer, frame);
    if (!rafScheduled) {
      rafScheduled = true;
      requestAnimationFrame(flush);
    }
  }

  return { dispatchCoalescedFrame };
}

export function setupLazyMobile(deferred) {
  if (!isMobile()) {
    deferred.loadFaults();
    deferred.loadHistory();
    return;
  }

  const btn = document.createElement('button');
  btn.textContent = 'Load advanced controls';
  btn.style.cssText = [
    'position:fixed',
    'top:12px',
    'right:12px',
    'z-index:9999',
    'padding:8px 12px',
    'background:#1a1a2e',
    'color:#e0e0e0',
    'border:1px solid #444',
    'border-radius:6px',
    'cursor:pointer',
    'font-size:13px',
  ].join(';');

  btn.addEventListener('click', () => {
    btn.disabled = true;
    btn.textContent = 'Loading…';
    Promise.all([deferred.loadFaults(), deferred.loadHistory()])
      .finally(() => btn.remove());
  });

  document.body.appendChild(btn);
}
