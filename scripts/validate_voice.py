"""Independent listening test: transcribe and judge generated Spanish speech.

Whisper screens clips during generation, so it cannot be the only judge of
whether a voice is right — it shares the teacher's blind spots. This asks
Mistral's Voxtral, which is not part of the pipeline, to do two things per clip:

* **transcribe** it, giving a second word error rate against the text that was
  supposed to be read, and
* **listen** to it and answer a fixed set of questions: accent, whether
  Castilian distinción is present, gender, pitch, naturalness, artefacts.

When reference recordings are given, the same questions are asked about the
reference once per voice and the answers are compared, which is a cheap speaker
check without a dedicated verification model.

    # teacher audio, before training
    python scripts/validate_voice.py --root data/pilot/audio --manifest data/pilot/audio/manifest.jsonl \\
        --references data/voices-es/references --limit 8 --output data/pilot/voice.json

    # a trained model: point --root and --manifest at the eval audio

Credentials: ``MISTRAL_API_KEY`` for the direct API (default), or
``OPENROUTER_API_KEY`` with ``--provider openrouter``.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ENDPOINTS = {"mistral": "https://api.mistral.ai/v1/chat/completions",
             "openrouter": "https://openrouter.ai/api/v1/chat/completions"}
MODELS = {"mistral": "voxtral-small-latest", "openrouter": "mistralai/voxtral-small-24b-2507"}

TRANSCRIBE_PROMPT = ("Transcribe this Spanish audio verbatim, exactly as spoken, including any errors. "
                     "Output only the transcription, with no quotes and no commentary.")

JUDGE_PROMPT = """Listen to this Spanish speech sample and answer with ONE JSON object and nothing else:
{
  "transcript": "<verbatim transcription>",
  "accent": "castilian" | "latin_american" | "mixed" | "unclear",
  "distincion": true | false | null,
  "gender": "female" | "male" | "unclear",
  "pitch": "low" | "medium" | "high",
  "timbre": "<three words, for example warm, bright, nasal>",
  "naturalness_1_10": <integer>,
  "intelligibility_1_10": <integer>,
  "artifacts": ["<short tag>", ...],
  "notes": "<one sentence>"
}

"distincion" means the speaker pronounces <z>, <ce> and <ci> as [θ] (as in most of
Spain) rather than [s]; use null when the clip has no such word. Be strict about
naturalness: 10 is indistinguishable from a human recording, 5 is clearly
synthetic but pleasant, 1 is unintelligible."""


def credentials(provider):
    variable = "MISTRAL_API_KEY" if provider == "mistral" else "OPENROUTER_API_KEY"
    key = os.environ.get(variable)
    if not key:
        raise SystemExit(f"{variable} is not set")
    return key


def call(provider, content, max_tokens=500, retries=4):
    body = {"model": MODELS[provider], "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}
    payload = json.dumps(body).encode()
    key = credentials(provider)
    for attempt in range(retries):
        request = urllib.request.Request(ENDPOINTS[provider], data=payload,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            response = json.load(urllib.request.urlopen(request, timeout=240))
            return response["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as error:
            # 500s from the audio endpoint are usually transient; 429 needs a longer wait.
            if error.code in (429, 500, 502, 503) and attempt + 1 < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:300]}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f"network: {error}") from error
    raise RuntimeError("giving up")


def audio_part(path):
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(Path(path).read_bytes()).decode(), "format": "wav"}}


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[start:end + 1])


def word_error(reference, hypothesis):
    import jiwer
    from tiny_voca.numbers_es import normalize
    return float(jiwer.wer(normalize(reference), normalize(hypothesis) or "<empty>"))


def describe(provider, path):
    return parse_json(call(provider, [audio_part(path), {"type": "text", "text": JUDGE_PROMPT}], max_tokens=600))


def compare(reference, generated):
    """What the reference and the generated clip have in common."""
    if not reference or not generated:
        return None
    return {"gender": reference.get("gender") == generated.get("gender"),
            "pitch": reference.get("pitch") == generated.get("pitch"),
            "accent": reference.get("accent") == generated.get("accent"),
            "reference_gender": reference.get("gender"), "reference_pitch": reference.get("pitch"),
            "reference_accent": reference.get("accent")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory the manifest's audio paths are relative to")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL with 'audio', 'text' and optionally 'voice_id'")
    source.add_argument("--report", type=Path, help="report.json from tiny_voca.evaluate")
    parser.add_argument("--references", type=Path, help="Directory with F2.wav / M2.wav")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--provider", default="mistral", choices=sorted(ENDPOINTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-judge", action="store_true", help="Transcribe only (cheaper)")
    args = parser.parse_args()
    credentials(args.provider)
    if args.report:
        # The eval report already pairs each generated clip with the text it read.
        report = json.loads(args.report.read_text(encoding="utf-8"))
        rows = [{"audio": sample["generated"], "text": sample["text"], "voice_id": sample["voice"]}
                for sample in report["samples"] if sample.get("generated")]
    else:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if (args.root / row["audio"]).is_file()][:args.limit]
    if not rows:
        raise SystemExit("no audio found for the manifest")
    described = {}
    results = []
    for row in rows:
        path = args.root / row["audio"]
        record = {"audio": str(path), "text": row["text"], "voice": row.get("voice_id")}
        try:
            transcript = call(args.provider, [audio_part(path), {"type": "text", "text": TRANSCRIBE_PROMPT}], max_tokens=300)
            record["transcript"] = transcript
            record["wer"] = round(word_error(row["text"], transcript), 4)
        except Exception as error:
            record["transcribe_error"] = str(error)
        if not args.no_judge:
            try:
                record.update(describe(args.provider, path))
            except Exception as error:
                record["judge_error"] = str(error)
            voice = row.get("voice_id")
            if args.references and voice:
                reference = args.references / f"{voice}.wav"
                if reference.is_file():
                    try:
                        if voice not in described:
                            described[voice] = describe(args.provider, reference)
                        record["reference_match"] = compare(described[voice], record)
                    except Exception as error:
                        record["reference_error"] = str(error)
        results.append(record)
        print(json.dumps({"audio": path.name, "wer": record.get("wer"), "naturalness": record.get("naturalness_1_10"),
                          "accent": record.get("accent"), "distincion": record.get("distincion"),
                          "match": (record.get("reference_match") or {}).get("gender")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"provider": args.provider, "model": MODELS[args.provider],
                                       "references": str(args.references) if args.references else None,
                                       "samples": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def mean(key):
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(values) / len(values), 2) if values else None
    summary = {"samples": len(results), "mean_wer": mean("wer"), "mean_naturalness": mean("naturalness_1_10"),
               "mean_intelligibility": mean("intelligibility_1_10"),
               "castilian": sum(r.get("accent") == "castilian" for r in results),
               "distincion": sum(r.get("distincion") is True for r in results),
               "gender_match": sum((r.get("reference_match") or {}).get("gender") is True for r in results),
               "errors": sum("transcribe_error" in r or "judge_error" in r for r in results)}
    print(json.dumps({"summary": summary}, indent=1))


if __name__ == "__main__":
    main()
