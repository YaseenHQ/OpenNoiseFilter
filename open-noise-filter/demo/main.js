// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

import { createNoiseFilter } from "../dist/index.mjs";

const $ = (id) => document.getElementById(id);
const statsEl = $("stats");

let ctx, media, filter, latestStats = null;

function render() {
  if (!filter) { statsEl.textContent = "not running"; return; }
  const s = latestStats;
  statsEl.textContent =
    `quality   ${filter.quality}\n` +
    `thread    ${filter.thread}\n` +
    `latency   ${(filter.latency * 1000).toFixed(1)} ms\n` +
    (s ? `underruns ${s.underruns}\noverruns  ${s.overruns}\nskips     ${s.skips}\ntarget    ${s.target}\n` : "") +
    `enabled   ${filter.enabled}`;
}

let rebuildChain = Promise.resolve();
function rebuild() {
  rebuildChain = rebuildChain.then(doRebuild).catch((e) => { statsEl.textContent = "error: " + e; });
  return rebuildChain;
}
async function doRebuild() {
  if (!ctx) return;
  const prev = filter;
  filter = await createNoiseFilter(ctx, {
    quality: $("quality").value,
    thread: $("thread").value || undefined,
    maxChannels: Number($("channels").value),
    enabled: !$("bypass").checked,
    onStats: (s) => { latestStats = s; render(); },
  });
  media.connect(filter.node);
  filter.node.connect(ctx.destination);
  if (prev) { try { media.disconnect(prev.node); } catch { /* already gone */ } prev.destroy(); }
  latestStats = null;
  render();
}

$("start").onclick = async () => {
  if (ctx) return;
  $("start").disabled = true;
  try {
    ctx = new AudioContext({ sampleRate: 48000 });
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    media = ctx.createMediaStreamSource(stream);
    await rebuild();
  } catch (e) {
    statsEl.textContent = "error: " + e;
    $("start").disabled = false;
  }
};
$("quality").onchange = rebuild;
$("thread").onchange = rebuild;
$("channels").onchange = rebuild;
$("bypass").onchange = () => { if (filter) { filter.setEnabled(!$("bypass").checked); render(); } };

// expose state for automated smoke tests
window.__demo = { get filter() { return filter; }, get stats() { return latestStats; } };
