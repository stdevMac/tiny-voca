# Training a Spanish tiny-voca

The English model cost $9.56 of GPU time: 24 hours on one rented RTX 4090, of
which 7.7 hours trained the released checkpoint. The Spanish run has the same
shape, so the budget is the same order of magnitude. The numbers below are
estimates from the local pilot, not measurements on the target card.

> **How the first run went.** This runbook was written before the run, and its
> estimates are kept as they were. The run itself used one RTX 4090. It read
> 11,050 sentences with both voices; 20,790 clips (24.5 h) passed screening and
> 19,238 utterances (22.7 h) went into the training split. Training ran 122,000
> steps, then a 20,000-step anneal at a low learning rate
> (`scripts/anneal_es.sh`); the released model is step 142,000. What it
> measured is in [round2.md](round2.md). The data is published as
> [marcosmaceo/tiny-voca-es](https://huggingface.co/datasets/marcosmaceo/tiny-voca-es)
> and the model as
> [marcosmaceo/tiny-voca-es](https://huggingface.co/marcosmaceo/tiny-voca-es).

## What is already proven

Run on this laptop (M3, 16 GB, no GPU), with macOS `say` standing in for the
teacher:

| Stage | Result |
|---|---|
| `scripts/build_text_es.py` | 300 sentences, 30 templates, no unknown phonemes |
| `scripts/generate_audio_local.py` | 240 clips, 2 voices, 24 kHz |
| `tiny_voca.prepare` | 240/240 accepted |
| `tiny_voca.annotate` | 192 train / 15 validation / 21 test accepted, 10.4 min of audio |
| `tiny_voca.train` | 200 steps, duration loss 1.28 → 0.32, pitch error 4.9 → 3.6 semitones |
| `tiny_voca.infer` | 2.6 s of 24 kHz audio, peak 0.40 |
| `tiny_voca.export` | 21.4 MB ONNX |

So the chain works in Spanish. What is *not* proven is the teacher: macOS voices
are not Qwen3-TTS, and the timing probe showed it (F2 `vowel_to_voiceless`
flip-found 0.39, against 0.77 for M2 and 0.58–0.64 for the English corpus). That
number is the first thing to check on real teacher audio.

## Cost

| Stage | Card | Hours | Cost |
|---|---|---|---|
| Voice design | RTX 4000 Ada | 0.2 | $0.10 |
| Teacher TTS, ~19 h of audio | RTX 4000 Ada | 4–8 | $2–4 |
| Features + alignment | RTX 4000 Ada | 2–4 | $1–2 |
| Training, 104k steps | RTX 4000 Ada | 14–20 | $7–10 |
| Failed attempts and re-runs | | | $5 |
| **Total** | | | **~$15–20** |

The card is a Linode/Akamai **RTX 4000 Ada x1 Small**: $0.52/h, 4 vCPU, 16 GB
RAM, 500 GB disk, 20 GB VRAM. That is enough for Qwen3-TTS 1.7B in bf16 (~4 GB
of weights) and for this 4.5M-parameter model. A 4090 is roughly twice as fast
per step; a Quadro RTX 6000 at $1.50/h is not worth it here.

A pilot that generates 30 minutes of audio and trains 5,000 steps costs about
$2 and tells you whether the voice and the alignment are right.

## Phase 0 — laptop

```bash
make venv test          # 96 tests (6 skipped); TINY_VOCA_LANGUAGE=en make test runs the English side
make pilot              # ~1 h on an M3, no GPU
```

The pilot needs the two Common Voice files under `data/common-voice/`:

```bash
mkdir -p data/common-voice
curl -o data/common-voice/sentence-collector.txt \
  https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/es/sentence-collector.txt
curl -o data/common-voice/wiki.es.txt \
  https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/es/wiki.es.txt
```

Both are CC0. Listen to `data/pilot/runs/pilot/inference.pt` with `make speak
RUN=pilot DATA=data/pilot` — it will be a mumble after 200 steps, but the
phonemes printed above the summary must be correct Spanish.

## Phase 1 — GPU host

Once the host is up, one command runs everything:

```bash
./scripts/run_pipeline.sh pilot     # ~200 clips, gates on the alignment probe
./scripts/run_pipeline.sh full      # 9,000 sentences, the real run
```

It writes `data/budget.json` from `DEADLINE_HOURS` (default 30), runs each stage
with its own log under `logs/`, stops at the first failure, and refuses to train
if `scripts/check_annotations.py` is unhappy. `--skip-audio` resumes from the
audio stage. The manual commands below are what it runs, if you want to drive
them by hand.


Any Ubuntu 24.04 machine with an NVIDIA card and the driver installed. On
Linode: create an *RTX 4000 Ada x1 Small* Linode from the Ubuntu 24.04 image,
then:

```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv espeak-ng ffmpeg git build-essential
git clone https://github.com/stdevMac/tiny-voca.git tiny-voca-es && cd tiny-voca-es
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[prepare,annotate,evaluate,export]"
uv pip install --python .venv/bin/python -U qwen-tts faster-whisper
nvidia-smi          # the driver must be there before you start paying attention
```

Set a budget before anything else. Both `generate_audio.py` and `train.py` read
it and stop cleanly; re-reading it every 30 seconds means you can extend or cut
a run by editing the file.

```bash
python -c "import json,time; print(json.dumps({'deadline': time.time()+8*3600}))" > data/budget.json
```

## Phase 2 — sentences and voices

```bash
make scripts            # data/scripts-es.jsonl: 9000 sentences, 3 splits
make voices             # data/voices-es/{references/F2.wav,M2.wav,voices.json}
```

Listen to the two reference clips before going further. Everything downstream
copies them, so a bad voice is 19 hours of bad audio. The descriptions in
`voices/es-ES.json` ask for Castilian with distinción; if the clip has seseo,
the `θ` tokens will never match the audio and the alignment will reject rows.

There is a listening test that does not need your ears, using Mistral's Voxtral
(an audio model that is not part of this pipeline):

```bash
export MISTRAL_API_KEY=...
python scripts/validate_voice.py --root data/voices-es --manifest data/voices-es/manifest.jsonl \
    --limit 2 --output data/voices-es/voice.json
```

It reports word error rate against the reference text, accent, whether
distinción is present, naturalness and artefacts. On the two designed voices it
returned WER 0.0, accent `castilian`, `distincion: true`, naturalness 8/10. Rerun
it on the trained model's output at the end — it is the same command with a
different `--root`/`--manifest`.

## Phase 3 — teacher audio

```bash
make audio              # data/audio-es/{audio/,manifest.jsonl,review.jsonl,progress.json}
```

Resumable through `state.sqlite`: rerun the same command after a stop and it
picks up where it left off. Watch `progress.json` for `accepted_audio_hours`.
The target is 17–19 hours across both voices, matching the English corpus.

Screening is Whisper at 15% normalized WER, one retry, peak normalisation to
0.95. On 4 vCPU that is the slow part, not the GPU.

## Phase 4 — features and alignment

```bash
make features           # data/features-es/
make annotate           # data/annotations-es/
```

Read `data/annotations-es/summary.json` before training. The numbers that
matter, against the English corpus:

| Metric | English | What to expect |
|---|---|---|
| accepted / total | 17,326 / 17,371 (99.7%) | ≥ 98% |
| `vowel_to_voiceless` within 2 frames | 81–86% | ≥ 80% |
| `fricative_to_vowel` within 2 frames | 86–89% | ≥ 85% |
| mean phone length | 7.0–7.5 frames | 6–8 |
| one-frame phones | 0.2% | < 1% |

If the timing probe is much worse, stop. It means the teacher and the front end
disagree (wrong accent, wrong `θ`, or a voice that does not match its
description), and training on it wastes the run.

## Phase 5 — training

```bash
make train STEPS=104000
```

Same recipe as English: 8,000 spectral-only steps, then the adversarial term at
a gradient ratio of 0.7, teacher forcing ramped to zero. ~0.5–0.7 s/step on the
RTX 4000 Ada, so 14–20 hours.

Watch `data/runs/es-1/metrics.jsonl`:

* `duration_length_ratio` should approach 1.0 (the model learns how long the
  sentence is). It was 0.92 after 100 steps in the pilot.
* `pitch_error_semitones` should fall below ~2.
* `validation_predicted_mel` should keep falling after the GAN starts. If it
  rises, the adversarial term is too strong — lower `--gan-ratio`.

The run is resumable: `--resume data/runs/es-1/last.pt` continues exactly.

## Phase 6 — evaluation and export

```bash
make evaluate           # data/eval/es-1/report.json + audio pairs
make export             # data/runs/es-1/tiny-voca-es.onnx
```

`evaluate` reports WER of synthesized speech transcribed by Whisper (the
`small` model by default; the published figures used
`--asr-model large-v3-turbo`, as `scripts/post_training_es.sh` does), plus
duration ratio, pause ratio and DTW mel distance against the recordings. It
also checks that re-running the front end on the raw text reproduces the
annotated tokens — a mismatch there means the front end changed under the
checkpoint.

The English model reached 0.6–0.9% WER on held-out and 0.0% on new sentences.
The Spanish release reached 2.00% (validation), 1.08% (test) and 0.0% (new
sentences) with large-v3-turbo; see [round2.md](round2.md).
Those sets are tiny; treat the number as a smoke alarm, not a benchmark. Listen
to `data/eval/es-1/audio/` before believing any of it.

## Phase 7 — run it on the laptop

```bash
scp <host>:tiny-voca-es/data/runs/es-1/inference.pt weights/tiny-voca-es.pt
make speak CKPT=weights/tiny-voca-es.pt TEXT="Hola, ¿cómo estás?" OUT=hola.wav
```

To use the released model instead of your own, download it with
`hf download marcosmaceo/tiny-voca-es --local-dir weights/es`.

`inference.pt` is ~18 MB and runs on CPU at roughly 70× real time on four
threads, as the English one does. The ONNX file is for the browser path.

## AWS instead of Linode

The `~/ops_infra/terraform_mltrain` stack is CPU-only (r7a/r7i/r6a for
LightGBM), so it cannot train this. And **this account currently has 0 GPU
quota**:

```
L-DB2E81BA  Running On-Demand G and VT instances   0.0
L-3819A6DF  All G and VT Spot Instance Requests    0.0
```

If you want AWS, request both (8 vCPU is enough) and then reuse the same
Terraform shape with `g6.xlarge` (L4, 24 GB, ~$0.70/h Spot) or `g5.xlarge`
(A10G). The bootstrap would need the NVIDIA driver and the CUDA build of
torch; the rest of the flow — Spot, SSM-only access, idle shutdown, S3 sync of
`runs/` — carries over unchanged. Until the quota is granted, Linode is the
faster path.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `unknown_phoneme` rejections | a word eSpeak cannot read; the corpus builder already filters these, so it means the scripts file changed |
| `vowels_not_voiced` rejections | the teacher's voice is breathy or the pitch tracker failed; check the clip |
| `preflight` fails on `cuda libraries` | cuDNN 9 is missing for CTranslate2; install `nvidia-cudnn-cu12` or run with `--asr-device cpu` |
| `preflight` fails on `espeak` | the front end did not produce `θ` for "Zaragoza": the eSpeak build is wrong |
| `weak_alignment` | teacher and front end disagree on pronunciation, usually accent (`θ` vs `s`) |
| `checkpoint was trained with a different phoneme inventory` | `TINY_VOCA_LANGUAGE` or `TINY_VOCA_ACCENT` differs from the one used to train |
| training loss flat at ~52 | the pilot's first steps; normal until the duration head learns |
