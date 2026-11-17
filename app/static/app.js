"use strict";

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  ["Greeting", "Hello! This is tiny voca, a small text to speech model running locally on your machine."],
  ["Numbers", "The meeting moved to 3:30 pm on March 14th, so we have about 45 minutes left. It costs $12.99."],
  ["Question", "Did you really think I would forget your birthday? Of course not!"],
  ["Story", "She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word."],
  ["Homographs", "I read the book you read yesterday. They live near the live music hall."],
];

const MAX_TEXT = 5000;

const state = {
  ready: false,
  busy: false,
  request: 0,
  controller: null,
  context: null,
  live: null,
  sampleRate: 24000,
  voice: "M2",
  samples: [],
  phonemes: [],
  peaks: null,
  duration: 0,
  url: null,
};

/* ---------- Setup ---------- */

for (const [label, text] of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip";
  button.textContent = label;
  button.addEventListener("click", () => {
    $("text").value = text;
    updateCount();
    update();
    $("text").focus();
  });
  $("examples").append(button);
}

$("text").addEventListener("input", () => {
  updateCount();
  update();
});

$("speed").addEventListener("input", () => {
  $("speed-value").textContent = `${Number($("speed").value).toFixed(2)}×`;
});

$("pitch").addEventListener("input", () => {
  const value = Number($("pitch").value);
  $("pitch-value").textContent = `${value > 0 ? "+" : ""}${value.toFixed(1)} st`;
});

$("pitch-range").addEventListener("input", () => {
  $("pitch-range-value").textContent = `${Number($("pitch-range").value).toFixed(2)}×`;
});

$("speak").addEventListener("click", speak);
$("play").addEventListener("click", togglePlay);

$("audio").addEventListener("play", () => {
  if (state.live) state.live.stop();
  updatePlayButton();
  tick();
});
$("audio").addEventListener("pause", updatePlayButton);
$("audio").addEventListener("ended", updatePlayButton);
$("audio").addEventListener("timeupdate", updateTime);
$("audio").addEventListener("loadedmetadata", updateTime);

$("wave-wrap").addEventListener("pointerdown", startSeek);
$("wave-wrap").addEventListener("pointermove", moveSeek);
$("wave-wrap").addEventListener("pointerup", endSeek);
$("wave-wrap").addEventListener("pointercancel", endSeek);

$("text").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("speak").disabled) $("speak").click();
  }
});

window.addEventListener("resize", () => {
  if (state.peaks) drawWave(progress());
});

const dark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
if (dark && dark.addEventListener) {
  dark.addEventListener("change", () => {
    if (state.peaks) drawWave(progress());
  });
}

updateCount();
update();
init();

/* ---------- Model connection ---------- */

async function init() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status !== "ready") {
      throw new Error(data.message || data.detail || `HTTP ${response.status}`);
    }
    state.ready = true;
    state.sampleRate = data.sampleRate || 24000;
    if (data.parameters) {
      $("fact-params").textContent = Number(data.parameters).toLocaleString("en-US");
    }
    if (data.loadMs) {
      $("fact-load").textContent = `${(data.loadMs / 1000).toFixed(2)} s`;
    }
    setStatus("Ready. Type something and press Speak.");
    update();
  } catch (error) {
    setStatus(`Could not start the model: ${error.message}`, null, true);
  }
}

/* ---------- Generation ---------- */

async function speak() {
  if (state.busy || (state.live && state.live.playing)) {
    stop();
    return;
  }
  const text = $("text").value.trim();
  if (!text || !state.ready) return;

  state.busy = true;
  state.request += 1;
  const id = state.request;
  state.controller = new AbortController();
  state.voice = selectedVoice();
  state.phonemes = [];
  state.samples = [];
  state.peaks = null;
  state.duration = 0;

  listen();
  update();
  setStatus("Generating…", 0);

  try {
    const response = await fetch("/api/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        voice: state.voice,
        speed: Number($("speed").value),
        pitch_semitones: Number($("pitch").value),
        pitch_range: Number($("pitch-range").value),
      }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join("; ")
        : data.detail;
      throw new Error(detail || data.message || `HTTP ${response.status}`);
    }
    await readEvents(response, id);
  } catch (error) {
    if (error.name === "AbortError" || id !== state.request) return;
    state.busy = false;
    state.controller = null;
    if (state.live) state.live.end();
    update();
    setStatus(error.message, null, true);
  }
}

function stop() {
  state.request += 1;
  if (state.controller) {
    state.controller.abort();
    state.controller = null;
  }
  if (state.live) state.live.stop();
  state.busy = false;
  update();
  setStatus("Stopped.");
}

async function readEvents(response, id) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) handleEvent(JSON.parse(line), id);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) handleEvent(JSON.parse(buffer), id);
}

function handleEvent(event, id) {
  if (id !== state.request) return;

  if (event.type === "start") {
    state.sampleRate = event.sampleRate || state.sampleRate;
    setStatus(`Speaking ${event.total} sentence${event.total === 1 ? "" : "s"}…`, 0);
  } else if (event.type === "sentence") {
    const samples = decodeAudio(event.audio);
    state.samples.push(samples);
    state.phonemes.push(event.phonemes);
    if (state.live) state.live.add(samples, event.sampleRate || state.sampleRate);
    setStatus(`Speaking sentence ${event.index + 1} of ${event.total}`, (event.index + 1) / event.total);
  } else if (event.type === "done") {
    finish(event);
  } else if (event.type === "error") {
    throw new Error(event.message);
  }
}

function finish(event) {
  state.busy = false;
  state.controller = null;
  const played = state.live ? state.live.end() : false;

  const samples = concat(state.samples);
  state.samples = samples;
  state.duration = samples.length / state.sampleRate;
  state.peaks = computePeaks(samples, 220);

  const blob = wav(samples, state.sampleRate);
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = URL.createObjectURL(blob);

  const audio = $("audio");
  audio.src = state.url;
  audio.load();
  $("download").href = state.url;
  $("download").download = `tiny-voca-${state.voice}-${timestamp()}.wav`;

  $("result").hidden = false;
  $("m-audio").textContent = `${state.duration.toFixed(1)} s`;
  $("m-time").textContent = `${(event.generationMs / 1000).toFixed(2)} s`;
  $("m-speed").textContent = `${(state.duration / (event.generationMs / 1000)).toFixed(1)}× real time`;
  $("m-phonemes").textContent = event.phonemeMs < 1 ? "<1 ms" : `${Math.round(event.phonemeMs)} ms`;
  $("phonemes").textContent = state.phonemes.join("\n");
  $("time").textContent = `0:00 / ${formatTime(state.duration)}`;

  drawWave(0);
  updatePlayButton();
  setStatus(`Done in ${(event.generationMs / 1000).toFixed(2)} s.`);
  update();

  // The sentences have been playing since the first one was made. If the
  // browser refused the live sound, play the whole clip now instead.
  if (!played) audio.play().catch(() => {});
}

/* ---------- Audio plumbing ---------- */

function listen() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!state.context && Context) {
    try {
      state.context = new Context({ sampleRate: state.sampleRate });
    } catch (error) {
      try {
        state.context = new Context();
      } catch (failed) {
        state.context = null;
      }
    }
  }
  if (!state.live && state.context) {
    state.live = livePlayer(state.context, update);
  }
  const audio = $("audio");
  if (!audio.paused) audio.pause();
  if (state.live) state.live.begin();
}

function decodeAudio(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function concat(chunks) {
  let length = 0;
  for (const chunk of chunks) length += chunk.length;
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  return samples;
}

function wav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/* ---------- Player ---------- */

function togglePlay() {
  const audio = $("audio");
  if (!audio.src) return;
  if (state.live && state.live.playing) state.live.stop();
  if (audio.paused) audio.play().catch(() => {});
  else audio.pause();
}

function updatePlayButton() {
  const playing = !$("audio").paused;
  $("play").classList.toggle("is-playing", playing);
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
}

let frame = 0;

function tick() {
  cancelAnimationFrame(frame);
  const audio = $("audio");
  if (audio.paused) return;
  drawWave(progress());
  updateTime();
  frame = requestAnimationFrame(tick);
}

function progress() {
  const audio = $("audio");
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    return audio.currentTime / audio.duration;
  }
  return state.duration ? audio.currentTime / state.duration : 0;
}

function updateTime() {
  const audio = $("audio");
  const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : state.duration;
  const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
  $("time").textContent = `${formatTime(current)} / ${formatTime(duration)}`;
}

let seeking = false;

function startSeek(event) {
  if (!state.peaks) return;
  seeking = true;
  if ($("wave-wrap").setPointerCapture) $("wave-wrap").setPointerCapture(event.pointerId);
  seekTo(event);
}

function moveSeek(event) {
  if (seeking) seekTo(event);
}

function endSeek(event) {
  if (!seeking) return;
  seeking = false;
  try {
    if ($("wave-wrap").releasePointerCapture) $("wave-wrap").releasePointerCapture(event.pointerId);
  } catch (error) {
    /* the pointer was already released */
  }
}

function seekTo(event) {
  const audio = $("audio");
  const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : state.duration;
  if (!duration) return;
  const rect = $("wave-wrap").getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  audio.currentTime = ratio * duration;
  drawWave(ratio);
  updateTime();
}

function computePeaks(samples, count) {
  const peaks = new Float32Array(count);
  if (!samples.length) return peaks;
  const block = Math.max(1, Math.floor(samples.length / count));
  for (let i = 0; i < count; i++) {
    const start = i * block;
    const end = Math.min(samples.length, start + block);
    let peak = 0;
    for (let j = start; j < end; j++) {
      const value = Math.abs(samples[j]);
      if (value > peak) peak = value;
    }
    peaks[i] = peak;
  }
  let loudest = 0;
  for (let i = 0; i < peaks.length; i++) {
    if (peaks[i] > loudest) loudest = peaks[i];
  }
  if (loudest > 0) {
    for (let i = 0; i < peaks.length; i++) peaks[i] /= loudest;
  }
  return peaks;
}

function drawWave(value = 0) {
  const canvas = $("wave");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const peaks = state.peaks;
  if (!peaks || !peaks.length) return;

  const styles = getComputedStyle(document.documentElement);
  const played = styles.getPropertyValue("--accent").trim() || "#b4421f";
  const rest = styles.getPropertyValue("--line-strong").trim() || "#c8bcaa";
  const bars = peaks.length;
  const step = width / bars;
  const barWidth = Math.max(1, step * 0.62);
  const middle = height / 2;
  const maxHeight = height * 0.92;

  for (let i = 0; i < bars; i++) {
    const barHeight = Math.max(2 * dpr, peaks[i] * maxHeight);
    const x = i * step + (step - barWidth) / 2;
    const y = middle - barHeight / 2;
    context.fillStyle = (i + 0.5) / bars < value ? played : rest;
    if (context.roundRect) {
      context.beginPath();
      context.roundRect(x, y, barWidth, barHeight, barWidth / 2);
      context.fill();
    } else {
      context.fillRect(x, y, barWidth, barHeight);
    }
  }
}

/* ---------- Small helpers ---------- */

function setStatus(text, value = null, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", Boolean(isError));
  const bar = $("progress");
  bar.hidden = value == null;
  if (value != null) {
    const percent = Math.round(value * 100);
    bar.firstElementChild.style.width = `${percent}%`;
    bar.setAttribute("aria-valuenow", String(percent));
  }
}

function update() {
  const live = Boolean(state.live && state.live.playing);
  const active = state.busy || live;
  $("speak-label").textContent = active ? "Stop" : "Speak";
  $("speak").classList.toggle("is-stop", active);
  $("speak").setAttribute("aria-label", active ? "Stop speaking" : "Speak the text");
  $("speak").disabled = !active && (!state.ready || !$("text").value.trim());
}

function updateCount() {
  const length = $("text").value.length;
  $("count").textContent = `${length.toLocaleString("en-US")} / ${MAX_TEXT.toLocaleString("en-US")}`;
  $("count").classList.toggle("near-limit", length > MAX_TEXT * 0.9);
}

function selectedVoice() {
  return document.querySelector('input[name="voice"]:checked').value;
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function timestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}
