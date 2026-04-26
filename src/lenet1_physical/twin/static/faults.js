export function setupFaultControls(rootEl) {
  const section = document.createElement('section');
  section.id = 'faults-panel';
  section.style.cssText = 'padding:8px;border:1px solid #555;margin:8px 0;font-size:13px;';

  const title = document.createElement('strong');
  title.textContent = 'Faults';
  section.appendChild(title);

  // --- Kill LED row ---
  const ledRow = document.createElement('div');
  ledRow.style.marginTop = '6px';

  const ledChainLabel = document.createElement('label');
  ledChainLabel.textContent = 'Kill LED — Chain: ';
  const deadChainSel = document.createElement('select');
  deadChainSel.id = 'fault-dead-chain';
  ledChainLabel.appendChild(deadChainSel);
  ledRow.appendChild(ledChainLabel);

  const ledPosLabel = document.createElement('label');
  ledPosLabel.style.marginLeft = '6px';
  ledPosLabel.textContent = 'Pos: ';
  const deadPosSel = document.createElement('select');
  deadPosSel.id = 'fault-dead-pos';
  ledPosLabel.appendChild(deadPosSel);
  ledRow.appendChild(ledPosLabel);

  const deadToggleBtn = document.createElement('button');
  deadToggleBtn.id = 'fault-dead-toggle';
  deadToggleBtn.style.marginLeft = '6px';
  deadToggleBtn.textContent = 'Kill LED';
  ledRow.appendChild(deadToggleBtn);

  const deadStatus = document.createElement('span');
  deadStatus.id = 'fault-dead-status';
  deadStatus.style.cssText = 'margin-left:6px;color:#aaa;';
  ledRow.appendChild(deadStatus);
  section.appendChild(ledRow);

  // --- Break Chain row ---
  const chainRow = document.createElement('div');
  chainRow.style.marginTop = '6px';

  const chainLabel = document.createElement('label');
  chainLabel.textContent = 'Break Chain: ';
  const chainSel = document.createElement('select');
  chainSel.id = 'fault-chain-select';
  chainLabel.appendChild(chainSel);
  chainRow.appendChild(chainLabel);

  const chainToggleBtn = document.createElement('button');
  chainToggleBtn.id = 'fault-chain-toggle';
  chainToggleBtn.style.marginLeft = '6px';
  chainToggleBtn.textContent = 'Break Chain';
  chainRow.appendChild(chainToggleBtn);

  const chainStatus = document.createElement('span');
  chainStatus.id = 'fault-chain-status';
  chainStatus.style.cssText = 'margin-left:6px;color:#aaa;';
  chainRow.appendChild(chainStatus);
  section.appendChild(chainRow);

  // --- Undervoltage row ---
  const uvRow = document.createElement('div');
  uvRow.style.marginTop = '6px';

  const uvLabel = document.createElement('label');
  uvLabel.textContent = 'Undervoltage: ';
  const uvSpan = document.createElement('span');
  uvSpan.id = 'fault-uv-label';
  uvSpan.textContent = '100%';
  uvLabel.appendChild(uvSpan);
  uvLabel.append(' ');

  const uvSlider = document.createElement('input');
  uvSlider.id = 'fault-uv-slider';
  uvSlider.type = 'range';
  uvSlider.min = '0';
  uvSlider.max = '100';
  uvSlider.value = '100';
  uvSlider.style.verticalAlign = 'middle';
  uvLabel.appendChild(uvSlider);
  uvRow.appendChild(uvLabel);
  section.appendChild(uvRow);

  // --- Clear All row ---
  const clearRow = document.createElement('div');
  clearRow.style.marginTop = '6px';
  const clearBtn = document.createElement('button');
  clearBtn.id = 'fault-clear-all';
  clearBtn.textContent = 'Clear All Faults';
  clearRow.appendChild(clearBtn);
  section.appendChild(clearRow);

  rootEl.appendChild(section);

  // --- Local state ---
  const killedLeds = new Set();
  const brokenChains = new Set();

  function ledKey(chain, pos) {
    return `${chain}:${pos}`;
  }

  async function populateChains() {
    try {
      const res = await fetch('/mapping');
      const data = await res.json();
      const chains = Array.isArray(data.chains) ? data.chains : [];
      const ids = chains.length > 0
        ? chains.map((c, i) => (typeof c === 'object' && c.id !== undefined ? c.id : i))
        : Array.from({ length: 17 }, (_, i) => i);

      [deadChainSel, chainSel].forEach(sel => {
        sel.textContent = '';
        ids.forEach(id => {
          const opt = document.createElement('option');
          opt.value = String(id);
          opt.textContent = String(id);
          sel.appendChild(opt);
        });
      });

      deadPosSel.textContent = '';
      for (let p = 0; p < 60; p++) {
        const opt = document.createElement('option');
        opt.value = String(p);
        opt.textContent = String(p);
        deadPosSel.appendChild(opt);
      }
    } catch (e) {
      console.warn('faults: could not fetch /mapping', e);
    }
  }

  function updateDeadBtn() {
    const chain = parseInt(deadChainSel.value, 10);
    const pos = parseInt(deadPosSel.value, 10);
    deadToggleBtn.textContent = killedLeds.has(ledKey(chain, pos)) ? 'Revive LED' : 'Kill LED';
  }

  deadToggleBtn.addEventListener('click', async () => {
    const chain = parseInt(deadChainSel.value, 10);
    const pos = parseInt(deadPosSel.value, 10);
    const key = ledKey(chain, pos);
    const active = !killedLeds.has(key);
    try {
      await fetch('/fault/dead-led', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chain, pos, active }),
      });
      if (active) {
        killedLeds.add(key);
        deadStatus.textContent = `LED ${chain}:${pos} killed`;
      } else {
        killedLeds.delete(key);
        deadStatus.textContent = `LED ${chain}:${pos} revived`;
      }
      updateDeadBtn();
    } catch (e) {
      deadStatus.textContent = 'Error';
      console.error(e);
    }
  });

  deadChainSel.addEventListener('change', updateDeadBtn);
  deadPosSel.addEventListener('change', updateDeadBtn);

  chainToggleBtn.addEventListener('click', async () => {
    const chain = parseInt(chainSel.value, 10);
    const active = !brokenChains.has(chain);
    try {
      await fetch('/fault/broken-chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chain, active }),
      });
      if (active) {
        brokenChains.add(chain);
        chainStatus.textContent = `Chain ${chain} broken`;
        chainToggleBtn.textContent = 'Restore Chain';
      } else {
        brokenChains.delete(chain);
        chainStatus.textContent = `Chain ${chain} restored`;
        chainToggleBtn.textContent = 'Break Chain';
      }
    } catch (e) {
      chainStatus.textContent = 'Error';
      console.error(e);
    }
  });

  chainSel.addEventListener('change', () => {
    const chain = parseInt(chainSel.value, 10);
    chainToggleBtn.textContent = brokenChains.has(chain) ? 'Restore Chain' : 'Break Chain';
  });

  uvSlider.addEventListener('input', async () => {
    const pct = parseInt(uvSlider.value, 10);
    uvSpan.textContent = `${pct}%`;
    try {
      await fetch('/fault/undervoltage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: pct / 100 }),
      });
    } catch (e) {
      console.error(e);
    }
  });

  clearBtn.addEventListener('click', async () => {
    try {
      await fetch('/fault/clear', { method: 'POST' });
      killedLeds.clear();
      brokenChains.clear();
      uvSlider.value = '100';
      uvSpan.textContent = '100%';
      deadStatus.textContent = '';
      chainStatus.textContent = '';
      deadToggleBtn.textContent = 'Kill LED';
      chainToggleBtn.textContent = 'Break Chain';
    } catch (e) {
      console.error(e);
    }
  });

  populateChains();
}
