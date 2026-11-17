// Runs the Spanish front end (eSpeak NG 1.52.0 compiled to WebAssembly, then
// site/frontend.js) and the model (ONNX Runtime Web) off the main thread.
// A module worker. Messages in: {type: "load", model} and
// {type: "speak", id, text, voice, speed, accent}. Messages out: status, ready,
// sentence (with its audio, as soon as it is made), done (with the whole clip),
// error (plain: true when the message is meant for the reader as it is).
//
// Speaking follows tiny_voca.infer.speak_text: sentences of the finished text,
// each cut after commas, then between words, while it is too long for the model.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
import { createEspeak } from "./espeak/espeak.js";
import { NothingToPronounce, phonemize, SYMBOLS } from "./frontend.js";
import { finished, sentences, shorter } from "./sentences.js";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
// The eSpeak NG version the model was trained with; another spells some words differently.
const ESPEAK_VERSION = "1.52.0";
// Shown when no sentence of a text could be read. tiny_voca.languages.spanish.ONLY_LANGUAGE says the same.
const ONLY_SPANISH = "tiny-voca solo lee español por ahora.";
// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;

let espeak, session, config;

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
// The key is the URL, which names a fixed revision of the model, so a new model is a new URL.
async function download(url, label) {
  let store = null;
  try {
    store = await caches.open("tiny-voca-es");
    const hit = await store.match(url);
    if (hit) { status(label, 1); return new Uint8Array(await hit.arrayBuffer()); }
  } catch (error) { store = null; }
  const bytes = await fetchBytes(url, label);
  if (store) store.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadFrontEnd() {
  espeak = await createEspeak(new URL("./espeak/", import.meta.url).href);
  if (espeak.version !== ESPEAK_VERSION) console.warn(`eSpeak NG ${espeak.version}, the model was trained with ${ESPEAK_VERSION}`);
  phonemize("Hola.", espeak.phonemize);  // warm up
}

async function loadModel(base) {
  const response = await fetch(base + "tiny-voca-es.json");
  if (!response.ok) throw new Error(`${base}tiny-voca-es.json: ${response.status}`);
  config = await response.json();
  if (JSON.stringify(config.symbols) !== JSON.stringify(SYMBOLS)) throw new Error("the model's symbols differ from the front end's");
  const bytes = await download(base + "tiny-voca-es.onnx", "Descargando el modelo");
  status("Iniciando el modelo");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  let frames = 0;
  for (const value of output.durations.data) frames += Number(value);
  return frames > config.max_frames ? null : output.audio.data;
}

function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// One piece of text -> its utterances, each [phones, audio], in order. Too long for one call of the
// model (max_tokens phonemes or max_frames frames): cut it smaller. Nothing to pronounce: nothing.
async function read(piece, voice, speed, accent, times) {
  let phones;
  let mark = performance.now();
  try {
    phones = phonemize(piece, espeak.phonemize, { accent });
  } catch (error) {
    if (error instanceof NothingToPronounce) return [];
    throw error;
  } finally {
    times.phonemes += performance.now() - mark;
  }
  if (phones.ids.length <= config.max_tokens - 12) {
    mark = performance.now();
    const audio = await synthesize(phones, voice, speed);
    times.model += performance.now() - mark;
    if (audio) return [[phones, audio]];
  }
  const pieces = shorter(piece);
  if (!pieces.length) throw new Error(`una palabra es demasiado larga: ${piece.slice(0, 40)}`);
  const result = [];
  for (const part of pieces) result.push(...await read(part, voice, speed, accent, times));
  return result;
}

const spelled = (phones) => phones.symbols.join(" ");

async function speak({ id, text, voice, speed, accent }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  const spoken = [];
  const times = { phonemes: 0, model: 0 };
  for (let index = 0; index < list.length; index++) {
    for (const [phones, audio] of await read(list[index], voice, speed, accent, times)) {
      chunks.push(audio);
      spoken.push(spelled(phones));
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: spelled(phones), audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: ONLY_SPANISH, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemes: spoken,
                phonemeTime: times.phonemes, modelTime: times.model, totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadFrontEnd(), loadModel(data.model)]);
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate, espeak: espeak.version });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
