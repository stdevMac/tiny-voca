#!/usr/bin/env python3
"""Build a self-contained blind listening page for clips that sit next to it.

The differences between synthesis variants are small, so the page makes them
easy to hear (switching clips keeps the playback position; the clips of a row
are sample-aligned), keeps the variant names out of sight while the listener
rates, and ends in a text the listener can paste back in one go. Candidates are
shuffled per row at build time, so the order is fixed in the HTML.

    python scripts/listening_page.py data/eval/heads --title "Word edges" --seed 3

reads DIR/manifest.jsonl (one clip per line: row, voice, text, role, variant,
description, file) and writes DIR/index.html. Name the wav files neutrally: the
file name is the one place a variant could leak (the browser may show it).
"""

import argparse
import hashlib
import html
import json
import random
import string
from pathlib import Path

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#e3e3e3;--card:#f6f6f6;--acc:#0b5fff;--accbg:#e4edff}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mute:#9a9a9a;--line:#333;--card:#202020;--acc:#7aa9ff;--accbg:#1c2a46}}
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;margin:0 auto;padding:0 1rem 3rem;background:var(--bg);color:var(--fg)}
h1{font-size:1.4rem;margin:1.5rem 0 .5rem}h2{font-size:1.1rem;margin:2rem 0 .5rem;border-bottom:1px solid var(--line)}
p{margin:.5rem 0}.mute,.howto{color:var(--mute)}.howto{font-size:.9rem}
.bar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:.5rem 1rem;align-items:center;background:var(--bg);padding:.6rem 0;border-bottom:1px solid var(--line);margin-top:1rem}
.bar progress{width:10rem;max-width:30vw}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.25rem .7rem;cursor:pointer}
button:hover{border-color:var(--acc)}button:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.row{border:1px solid var(--line);border-radius:8px;padding:.7rem .9rem;margin:.9rem 0}.row.active{border-color:var(--acc)}
.sentence{font-size:1.05rem;margin:0 0 .6rem}.sentence small{color:var(--mute);font-size:.75rem;margin-left:.5rem}
.clip{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .8rem;padding:.3rem .5rem;border-left:4px solid transparent;border-radius:4px;margin:.15rem 0}
.clip.playing{background:var(--accbg);border-left-color:var(--acc);font-weight:600}
.play{min-width:8rem;text-align:left}.clip.playing .play{border-color:var(--acc)}.ic{display:inline-block;width:1.2em}
.rate{border:0;margin:0;padding:0;display:flex;gap:.7rem;align-items:center;font-weight:400}.rate label{cursor:pointer}
.rate legend,.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.reveal{font-size:.85rem;font-weight:400;color:var(--acc)}.broken .play{text-decoration:line-through}
.transport{display:flex;gap:.7rem;align-items:center;margin:.5rem 0}.transport input{flex:1;min-width:0}.time{font-size:.8rem;color:var(--mute);min-width:6.5rem;text-align:right}
textarea{width:100%;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.3rem .5rem}
.res textarea{font:12px/1.4 ui-monospace,Menlo,monospace;min-height:16rem}.res{margin-top:2rem}
"""

JS = r"""
(function () {
"use strict";
var DATA = JSON.parse(document.getElementById("data").textContent);
var KEY = "listening:" + DATA.key;
var VALUES = ["worse", "same", "better"];
var state = {ratings: {}, notes: {}};
var rows = [], active = null, raf = 0;

function rkey(row, variant) { return JSON.stringify([row, variant]); }
function store(write) {
  try {
    if (write) localStorage.setItem(KEY, JSON.stringify(state));
    else return localStorage.getItem(KEY);
  } catch (e) {}
  return null;
}
function restore() {
  try {
    var o = JSON.parse(store(false) || "null");
    if (!o) return;
    DATA.rows.forEach(function (r) {
      if (typeof o.notes[r.row] === "string") state.notes[r.row] = o.notes[r.row];
      r.clips.forEach(function (c) {
        var v = o.ratings[rkey(r.row, c.variant)];
        if (c.kind === "candidate" && VALUES.indexOf(v) >= 0) state.ratings[rkey(r.row, c.variant)] = v;
      });
    });
  } catch (e) { state = {ratings: {}, notes: {}}; }
}
function fmt(t) { return (isFinite(t) ? t : 0).toFixed(1); }

function livePos(r) {
  var c = r.cur;
  if (!c || c.teacher) return 0;
  if (c.audio.ended) return 0;
  return c.audio.readyState >= 1 ? c.audio.currentTime : r.pos;
}
function seekClip(c, t) {
  if (c.audio.readyState >= 1) { try { c.audio.currentTime = t; } catch (e) {} c.pending = null; }
  else c.pending = t;
}
function prime(c) { c.audio.preload = "auto"; c.audio.load(); }
function stopAll(except) {
  rows.forEach(function (r) { r.clips.forEach(function (c) { if (c !== except && !c.audio.paused) c.audio.pause(); }); });
}
function start(r, c, t) {
  setActive(r);
  stopAll(c);
  r.cur = c;
  seekClip(c, t);
  if (!c.teacher) r.pos = t;
  var p = c.audio.play();
  if (p && p.catch) p.catch(refresh);
  if (!r.loaded) {                       // first play: warm the siblings so switching is instant
    r.loaded = true;
    r.clips.forEach(function (o) { if (o !== c) prime(o); });
  }
  refresh();
}
function toggle(r, c) {
  setActive(r);
  if (!c.audio.paused) { c.audio.pause(); return; }
  var t = c.teacher ? (r.cur === c && !c.audio.ended ? c.audio.currentTime : 0) : livePos(r);
  start(r, c, t);
}
function fromStart(r) { start(r, r.cur || r.shipped, 0); }
function seekFrac(r, f) {
  var c = r.cur || r.shipped, d = c.audio.duration;
  r.cur = c;
  if (!isFinite(d)) { r.want = f; prime(c); return; }
  seekClip(c, f * d);
  if (!c.teacher) r.pos = f * d;
  updateBar(r);
}
function updateBar(r) {
  var c = r.cur, d = c ? c.audio.duration : NaN;
  var t = !c ? 0 : (c.audio.readyState >= 1 ? c.audio.currentTime : (c.teacher ? 0 : r.pos));
  if (!r.drag) r.bar.value = isFinite(d) && d > 0 ? Math.round(t / d * 1000) : 0;
  r.time.textContent = fmt(t) + " / " + (isFinite(d) ? fmt(d) : "-") + " s";
}
function refresh() {
  var any = false;
  rows.forEach(function (r) {
    r.clips.forEach(function (c) {
      var on = !c.audio.paused;
      any = any || on;
      c.item.classList.toggle("playing", on);
      c.btn.setAttribute("aria-pressed", on ? "true" : "false");
      c.ic.textContent = on ? "⏸" : "▶";
    });
    updateBar(r);
  });
  if (any && !raf) raf = requestAnimationFrame(tick);
}
function tick() {
  raf = 0;
  var any = false;
  rows.forEach(function (r) {
    if (r.cur && !r.cur.audio.paused) { any = true; updateBar(r); }
  });
  if (any) raf = requestAnimationFrame(tick);
}
function setActive(r) {
  active = r;
  rows.forEach(function (o) { o.el.classList.toggle("active", o === r); });
}

function build() {
  [].forEach.call(document.querySelectorAll(".row"), function (el, ri) {
    var meta = DATA.rows[ri];
    var r = {el: el, meta: meta, cur: null, pos: 0, want: null, loaded: false, drag: false, clips: []};
    r.bar = el.querySelector(".bar-seek");
    r.time = el.querySelector(".time");
    [].forEach.call(el.querySelectorAll(".clip"), function (item, ci) {
      var info = meta.clips[ci];
      var c = {item: item, info: info, audio: item.querySelector("audio"), btn: item.querySelector(".play"),
               ic: item.querySelector(".ic"), teacher: info.kind === "teacher", pending: null,
               radios: item.querySelectorAll("input[type=radio]"), reveal: item.querySelector(".reveal")};
      r.clips.push(c);
      if (info.kind === "shipped") r.shipped = c;
      c.btn.addEventListener("click", function () { toggle(r, c); });
      c.audio.addEventListener("play", function () { stopAll(c); r.cur = c; refresh(); });
      c.audio.addEventListener("pause", function () {
        if (r.cur === c && !c.teacher && c.audio.readyState >= 1 && !c.audio.ended) r.pos = c.audio.currentTime;
        refresh();
      });
      c.audio.addEventListener("ended", function () { if (!c.teacher) r.pos = 0; refresh(); });
      c.audio.addEventListener("timeupdate", function () { if (r.cur === c) updateBar(r); });
      c.audio.addEventListener("loadedmetadata", function () {
        if (c.pending !== null) seekClip(c, c.pending);
        if (r.want !== null && r.cur === c) {
          var f = r.want; r.want = null; seekFrac(r, f);
        }
        updateBar(r);
      });
      c.audio.addEventListener("error", function () { item.classList.add("broken"); });
      [].forEach.call(c.radios, function (inp) {
        inp.checked = state.ratings[rkey(meta.row, info.variant)] === inp.value;
        inp.addEventListener("change", function () {
          state.ratings[rkey(meta.row, info.variant)] = inp.value;
          store(true); summarize();
        });
      });
    });
    r.bar.addEventListener("pointerdown", function () { r.drag = true; });
    ["pointerup", "pointercancel", "blur"].forEach(function (n) { r.bar.addEventListener(n, function () { r.drag = false; }); });
    r.bar.addEventListener("input", function () { seekFrac(r, r.bar.value / 1000); });
    el.querySelector(".from-start").addEventListener("click", function () { fromStart(r); });
    var note = el.querySelector("textarea");
    note.value = state.notes[meta.row] || "";
    note.addEventListener("input", function () {
      if (note.value) state.notes[meta.row] = note.value; else delete state.notes[meta.row];
      store(true); summarize();
    });
    el.addEventListener("pointerdown", function () { setActive(r); });
    el.addEventListener("focusin", function () { setActive(r); });
    rows.push(r);
  });
}

function results() {
  var total = 0, rated = 0, L = [], notes = [], tally = {}, voices = [];
  DATA.rows.forEach(function (r) {
    if (voices.indexOf(r.voice) < 0) { voices.push(r.voice); tally[r.voice] = {names: [], n: {}}; }
    var t = tally[r.voice];
    r.clips.forEach(function (c) {
      if (c.kind !== "candidate") return;
      total++;
      if (t.names.indexOf(c.variant) < 0) { t.names.push(c.variant); t.n[c.variant] = {better: 0, same: 0, worse: 0}; }
      var v = state.ratings[rkey(r.row, c.variant)];
      if (!v) return;
      rated++;
      t.n[c.variant][v]++;
      L.push([r.row, c.letter, c.variant, v].join(" | "));
    });
    if (state.notes[r.row]) notes.push(r.row + " (" + r.voice + "): " + state.notes[r.row].replace(/\s*\n\s*/g, " / "));
  });
  var out = [DATA.title, rated + " of " + total + " rated; each rating is relative to Ships today", "",
             "row | letter | variant | rating"].concat(L.length ? L : ["(none yet)"]);
  out.push("", "Notes");
  out = out.concat(notes.length ? notes : ["(none)"]);
  out.push("", "Tally per variant (better / same / worse)");
  voices.forEach(function (v) {
    out.push(v);
    tally[v].names.forEach(function (n) {
      var k = tally[v].n[n];
      out.push("  " + n + ": " + k.better + " better, " + k.same + " same, " + k.worse + " worse");
    });
  });
  return {text: out.join("\n"), rated: rated, total: total};
}
function summarize() {
  var s = results();
  document.getElementById("count").textContent = s.rated + " of " + s.total + " rated";
  document.getElementById("meter").max = s.total || 1;
  document.getElementById("meter").value = s.rated;
  document.getElementById("out").value = s.text;
}

function setupBar() {
  var rev = document.getElementById("reveal");
  rev.checked = false;
  rev.addEventListener("change", function () {
    rows.forEach(function (r) {
      r.clips.forEach(function (c) {
        if (c.reveal) c.reveal.textContent = rev.checked ? c.info.variant + (c.info.description ? " — " + c.info.description : "") : "";
      });
    });
  });
  var ask = document.getElementById("clear"), sure = document.getElementById("sure"), timer = 0;
  function reset() { clearTimeout(timer); sure.hidden = true; ask.hidden = false; }
  ask.addEventListener("click", function () {
    ask.hidden = true; sure.hidden = false;
    timer = setTimeout(reset, 6000);
  });
  document.getElementById("keep").addEventListener("click", reset);
  document.getElementById("yes").addEventListener("click", function () {
    state = {ratings: {}, notes: {}};
    store(true);
    rows.forEach(function (r) {
      r.el.querySelector("textarea").value = "";
      r.clips.forEach(function (c) { [].forEach.call(c.radios, function (i) { i.checked = false; }); });
    });
    reset(); summarize();
  });
  var out = document.getElementById("out"), msg = document.getElementById("copied");
  document.getElementById("copy").addEventListener("click", function () {
    function fallback() {
      document.getElementById("resbox").open = true;
      out.focus(); out.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) {}
      msg.textContent = ok ? "Copied." : "Selected below; press Cmd/Ctrl+C.";
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(out.value).then(function () { msg.textContent = "Copied."; }, fallback);
    } else fallback();
  });
}

document.addEventListener("keydown", function (e) {
  if (!active || e.ctrlKey || e.metaKey || e.altKey) return;
  var t = e.target, tag = t && t.tagName;
  if (tag === "TEXTAREA" || (tag === "INPUT" && t.type !== "radio" && t.type !== "range" && t.type !== "checkbox")) return;
  var r = active, k = e.key, c = null, cands = r.clips.filter(function (x) { return x.info.kind === "candidate"; });
  if (k === " ") {
    if (tag === "BUTTON" || tag === "INPUT" || tag === "SUMMARY") return;
    c = r.cur || r.shipped;
  } else if (k === "0") c = r.shipped;
  else if (k.length === 1 && k.toLowerCase() === "t") c = r.clips[0].teacher ? r.clips[0] : null;
  else if (/^[1-9]$/.test(k)) c = cands[+k - 1];
  else if (/^[a-z]$/i.test(k)) c = cands[k.toUpperCase().charCodeAt(0) - 65];
  if (!c) return;
  e.preventDefault();
  toggle(r, c);
});

restore();
build();
setupBar();
summarize();
refresh();
})();
"""


def build(directory, title, intro, clips, seed=0):
    """Write directory/index.html and return its path."""
    directory = Path(directory)
    clips = [dict(c) for c in clips]
    key = hashlib.sha1(json.dumps([seed, clips], sort_keys=True).encode()).hexdigest()[:12]
    grouped = {}
    for clip in clips:
        grouped.setdefault(clip["row"], []).append(clip)
    rows = []
    for row, group in grouped.items():
        by_role = {role: [c for c in group if c["role"] == role] for role in ("teacher", "shipped", "candidate")}
        if len(by_role["shipped"]) != 1 or len(by_role["teacher"]) > 1 or len(group) != sum(map(len, by_role.values())):
            raise ValueError(f"row {row!r}: needs exactly one shipped, at most one teacher, and only known roles")
        if len(by_role["candidate"]) > len(string.ascii_uppercase):
            raise ValueError(f"row {row!r}: too many candidates to letter")
        for clip in group:
            if not (directory / clip["file"]).is_file():
                raise ValueError(f"row {row!r}: missing {directory / clip['file']}")
        # Decided here, not in the page: the order is fixed in the HTML and stable across reloads.
        candidates = by_role["candidate"][:]
        random.Random(f"{seed}/{row}").shuffle(candidates)
        for letter, clip in zip(string.ascii_uppercase, candidates):
            clip["_letter"] = letter
        ordered = by_role["teacher"] + by_role["shipped"] + candidates
        rows.append({"row": row, "voice": group[0]["voice"], "text": group[0]["text"], "clips": ordered})

    data = {"title": title, "rows": [], "key": key}
    voices = {}
    for index, row in enumerate(rows):
        data["rows"].append({"row": row["row"], "voice": row["voice"], "clips": [
            {"kind": "candidate" if "_letter" in c else c["role"], "letter": c.get("_letter", ""),
             "variant": c["variant"], "description": c["description"]} for c in row["clips"]]})
        voices.setdefault(row["voice"], []).append(index)

    body = []
    for voice, indices in voices.items():
        body.append(f"<h2>Voice {html.escape(voice)}</h2>")
        for index in indices:
            row = rows[index]
            items = []
            for ci, clip in enumerate(row["clips"]):
                letter = clip.get("_letter")
                label = letter or ("Teacher" if clip["role"] == "teacher" else "Ships today")
                rate = ""
                if letter:
                    radios = "".join(f"<label><input type='radio' name='rate-{index}-{ci}' value='{v}'> {v}</label>"
                                     for v in ("worse", "same", "better"))
                    rate = (f"<span class='reveal'></span><fieldset class='rate'><legend>Rating of {letter} compared with Ships today</legend>"
                            f"{radios}<span class='mute'>than Ships today</span></fieldset>")
                items.append(f"<div class='clip'><button type='button' class='play' aria-pressed='false' aria-label='Play {html.escape(label)}'>"
                             f"<span class='ic' aria-hidden='true'>▶</span>{html.escape(label)}</button>"
                             f"<audio preload='none' src='{html.escape(clip['file'])}'></audio>{rate}</div>")
            body.append(
                f"<section class='row' id='r{index}'><p class='sentence' lang='es'>{html.escape(row['text'])}"
                f"<small>{html.escape(str(row['row']))}</small></p>{''.join(items)}"
                f"<div class='transport'><input type='range' class='bar-seek' min='0' max='1000' step='1' value='0' aria-label='Position'>"
                f"<span class='time' aria-hidden='true'></span><button type='button' class='from-start'>From start</button></div>"
                f"<textarea rows='2' placeholder='Note (optional)' aria-label='Note for this sentence'></textarea></section>")

    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
{intro}
<p class='howto'>Click a clip to play it. While one plays, click another in the same row to hear it from the same moment; click the playing one to pause.
The teacher is a different recording, so it starts from the beginning, and so does the clip played after it.
Rate each lettered clip against <b>Ships today</b>. Keys, with a row selected: 0 = Ships today, T = Teacher, letters or 1, 2, 3 = candidates, space = pause.</p>
<div class='bar'><b id='count'></b><progress id='meter' value='0' max='1'></progress>
<label><input type='checkbox' id='reveal'> Reveal variants</label>
<span><button type='button' id='clear'>Clear ratings</button><span id='sure' hidden>Clear all ratings and notes? <button type='button' id='yes'>Yes, clear</button> <button type='button' id='keep'>Keep</button></span></span></div>
{''.join(body)}
<section class='res'><h2>Results</h2><p><button type='button' id='copy'>Copy results</button> <span id='copied' class='mute' role='status'></span></p>
<details id='resbox'><summary>Show the results text (it names the variants)</summary><textarea id='out' readonly></textarea></details></section>
<script type='application/json' id='data'>{payload}</script>
<script>{JS}</script></body></html>
"""
    path = directory / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--title", default="Blind listening comparison")
    parser.add_argument("--intro", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lines = (args.directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    clips = [json.loads(line) for line in lines if line.strip()]
    print(build(args.directory, args.title, args.intro, clips, args.seed))


if __name__ == "__main__":
    main()
