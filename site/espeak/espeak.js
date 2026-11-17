// eSpeak NG 1.52.0 (WebAssembly) phonemizer for Spanish.
//
//   import { createEspeak } from "./espeak/espeak.js";
//   const espeak = await createEspeak();          // or createEspeak(baseUrl)
//   espeak.phonemize("Hola mundo")                // "ˈo_l_a m_ˈu_n_d_o\n"
//
// phonemize(text) returns exactly what
//   espeak-ng -q --ipa --sep=_ -v es --stdin
// prints to stdout when its standard input is text + "\n".
// Works in a browser (window or module Worker) and in Node.

import createEspeakCore from "./espeak-ng-core.mjs";

const DATA_DIR = "/espeak-ng-data";
const VOICE = "es";
const isNode = typeof process === "object" && !!process.versions?.node
  && typeof window === "undefined" && typeof WorkerGlobalScope === "undefined";

async function resolveBase(baseUrl) {
  if (baseUrl == null) return new URL("./", import.meta.url);
  if (baseUrl instanceof URL) return baseUrl;
  let base = String(baseUrl);
  if (!base.endsWith("/")) base += "/";
  if (isNode && !/^[a-z][a-z0-9+.-]*:/i.test(base)) {
    const { pathToFileURL } = await import("node:url");
    return pathToFileURL(base);
  }
  return new URL(base, globalThis.location?.href ?? import.meta.url);
}

async function readBytes(url) {
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return new Uint8Array(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`espeak: ${response.status} fetching ${url}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes; // already inflated
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// espeak-ng-data.pack: gzip of "TVPK" | u32le n | JSON [[path, size], ...] (n bytes) | file bytes
function unpack(bytes) {
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== "TVPK") throw new Error("espeak: bad data pack");
  const n = new DataView(bytes.buffer, bytes.byteOffset).getUint32(4, true);
  const index = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  let offset = 8 + n;
  return index.map(([path, size]) => {
    const data = bytes.subarray(offset, offset + size);
    offset += size;
    return [path, data];
  });
}

export async function createEspeak(baseUrl) {
  const base = await resolveBase(baseUrl);
  const [wasmBinary, pack] = await Promise.all([
    readBytes(new URL("espeak-ng-core.wasm", base)),
    readBytes(new URL("espeak-ng-data.pack", base)).then(gunzip),
  ]);

  const stderr = [];
  const core = await createEspeakCore({
    wasmBinary,
    noInitialRun: true,
    print: () => {},
    printErr: (line) => stderr.push(line),
  });

  for (const [path, data] of unpack(pack)) {
    const full = `${DATA_DIR}/${path}`;
    core.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    core.FS.writeFile(full, data);
  }

  const withCString = (text, fn) => {
    const size = core.lengthBytesUTF8(text) + 1;
    const ptr = core._malloc(size);
    core.stringToUTF8(text, ptr, size);
    try { return fn(ptr); } finally { core._free(ptr); }
  };
  const status = withCString(DATA_DIR, (path) => withCString(VOICE, (voice) => core._tv_init(path, voice)));
  if (status !== 0) {
    throw new Error(`espeak: initialisation failed (status ${status}) ${stderr.join("\n")}`);
  }
  const version = core.UTF8ToString(core._tv_version());

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  function phonemize(text) {
    // The bytes the CLI would read from standard input.
    const input = encoder.encode(String(text) + "\n");
    const inputPtr = core._malloc(input.length + 1);
    const lengthPtr = core._malloc(4);
    try {
      core.HEAPU8.set(input, inputPtr);
      const outPtr = core._tv_phonemize(inputPtr, input.length, lengthPtr);
      if (!outPtr) throw new Error("espeak: phonemize failed");
      const length = core.HEAP32[lengthPtr >> 2];
      const output = decoder.decode(core.HEAPU8.slice(outPtr, outPtr + length));
      core._free(outPtr);
      return output;
    } finally {
      core._free(inputPtr);
      core._free(lengthPtr);
    }
  }

  return { phonemize, version };
}
