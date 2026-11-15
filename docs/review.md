# Adversarial review before the first real run

Written after the local pilot, before renting a GPU. Each finding has a
severity, the evidence, and what was changed. The point is to find the things
that would waste money at hour six, not to list everything that could be nicer.

## Blocking

### B1. `prepare` is single-threaded and pitch tracking dominates

Measured: 10 clips in 12.7 s on an M3, so **1.27 s per clip**. The corpus is
~16,700 training clips, so the feature step alone would take **5.9 hours** on
one core, and the three splits do not overlap because `prepare` holds a lock
per split.

`librosa.pyin` is the cost: it runs YIN at every candidate period and then does
a Viterbi pass. It is the right tracker for this recipe (the English model was
trained on it), so it stays — but it has to run in parallel.

**Fix:** `prepare --workers N`, a process pool around the per-row work. The
journal stays sequential and ordered, so resumption is unchanged. On the 8-vCPU
plan that is ~45 minutes instead of 6 hours.

### B2. Whisper screening runs serially on the CPU

`generate_audio.py` submits `analyze()` to a one-worker thread pool, so
transcription is serial and pinned to the CPU while the GPU idles. On 4–8 vCPU
that is 5–10 hours for 20 hours of audio, and it is the longest single stage
after training.

**Fix:** `--asr-device cuda`. On CUDA the work runs synchronously in the main
thread (no cross-thread CUDA context), which trades the overlap for roughly a
10× speedup. CPU stays the default so nothing changes for laptops.

### B3. `train.py` preloads the whole corpus into RAM

`train.preload(pool)` reads every feature file up front. At ~127 KB per second
of audio and ~19 hours of corpus that is **~8.7 GB**, plus torch, the CUDA
context and the optimizer. The 16 GB plan would swap or die; the failure would
happen *after* audio generation and alignment, i.e. after the money is spent.

**Fix:** use the 32 GB plan (`g2-gpu-rtx4000a1-m`, $0.67/h), and add a memory
estimate to `train.py` that falls back to lazy loading with a warning instead of
being killed by the OOM killer.

### B4. No preflight, so environment problems surface hours in

The run depends on: the NVIDIA driver, a CUDA build of torch, cuDNN 9 for
CTranslate2, `espeak-ng`, the wav2vec2 aligner in the HF cache, `qwen-tts`, and
enough disk. Any one missing turns into a failed stage after provisioning.

**Fix:** `scripts/preflight.py` checks all of it, downloads the models, and runs
three sentences through phonemize → prepare → annotate → train-one-step →
infer. It exits non-zero on the first problem. `run_pipeline.sh` calls it first.

### B5. No orchestration

The run is eight stages with different arguments, a budget file, and resumable
state. Doing it by hand at 3 a.m. is how runs get lost.

**Fix:** `scripts/run_pipeline.sh` runs the stages in order, logs to
`logs/<stage>.log`, writes the budget deadline, stops on the first failure, and
prints a one-line status per stage.

## High

### H1. The teacher may not have distinción, and that would poison everything

The `es-ES` front end emits `θ` for ⟨z⟩, ⟨ce⟩, ⟨ci⟩. If the designed voice is
seseante, every one of those tokens has no acoustic counterpart, the CTC
alignment weakens, and rows get rejected as `weak_alignment` — or worse, pass
with bad durations and teach the model a lisp it does not have.

**Fix:** the pilot is a gate, not a formality. It generates ~200 clips with the
real teacher, aligns them, and the pipeline refuses to continue unless the
timing probe and the rejection rate clear the thresholds in
`docs/training-es.md`. The probe is also the only cheap way to catch a voice
that does not match its description.

### H2. eSpeak version drift between the host and the laptop

The corpus and the annotations are produced on the host (Ubuntu 24.04 ships
espeak-ng 1.51); inference happens on the laptop (Homebrew ships 1.52). The
front-end fingerprint covers our inventory and rules but **not** the eSpeak
build, so a difference would silently produce different phones at inference
than at training.

**Fix:** `scripts/espeak_probe.py` phonemizes a fixed 40-sentence set and prints
a hash. Measured on a throwaway Ubuntu 24.04 box: 1.51 gives `89a0beaf…`, the
macOS 1.52 build gives `53fca553…`, and **1 of 40 sentences differs** — 1.51
writes "informe" as `i m f o ɾ m e` and 1.52 as `i n f o ɾ m e`. That is the
`n → m` assimilation before a labiodental, so it affects a whole word class
(`informe`, `confianza`, `enfermo`, `infinito`, …), not one word.

Building 1.52.0 from the git tag on the host reproduces the macOS hash exactly,
so `scripts/host_setup.sh` does that and the drift is gone rather than merely
detected.

### H3. Nothing guarantees the machine gets deleted

An hourly GPU instance left running is the classic way to turn a $20 run into a
$200 one.

**Fix:** three layers. `run_pipeline.sh` writes a hard deadline into
`data/budget.json`, which `train.py` and `generate_audio.py` already honour;
the script runs `linode-cli linodes delete` in a trap on exit; and the runbook
says to check `linode-cli linodes list` afterwards. The deadline is set from the
measured stage times, not guessed.

### H4. A crash inside `generate_audio` loses the batch

`generate()` catches `torch.cuda.OutOfMemoryError` and splits the batch, but any
other exception (a tokenizer edge case, a NaN waveform, a CUDA error) propagates
and kills the run. With 2,250 batches that is a real risk.

**Fix:** catch per batch, record the batch as failed in SQLite, and continue.
The retry pass picks those up. A run that loses 8 clips is fine; a run that
stops at batch 1,400 is not.

## Medium

### M1. `prepare` duplicate detection is order-dependent

`seen` is built from committed rows and updated as rows are accepted. In a
parallel run two identical ids could both pass. **Fix:** reject duplicates
against the whole pending set up front, before the pool starts.

### M2. Rejected clips still write their FLAC

`analyze()` writes the audio before deciding, so review clips take disk. It is
deliberate (you want to listen to failures) and 500 GB is enough. **No change**,
noted so it is not mistaken for a leak.

### M3. The pilot and the full run must not share state

Both use `state.sqlite` keyed by clip id. The pilot's ids are a subset of the
full run's, so pointing the full run at the pilot directory would silently skip
those clips. **Fix:** the pipeline uses separate roots and the runbook says so.

### M4. `evaluate` compares accented text with Whisper output

Whisper usually writes accents, but not always. A missing accent would count as
a word error. **Fix:** none for now — the number is a smoke alarm, not a
benchmark, and stripping accents would hide real pronunciation errors. Noted.

### M5. ONNX export is not verified on the host

`export` writes the file but nothing runs it. **Fix:** `scripts/preflight.py`
runs the ONNX file through `onnxruntime` and compares the waveform length with
the PyTorch path.

### M6. `--limit` picks the shortest sentences

The pilot uses the 200 shortest scripts, which under-represents long sentences
and the number templates. That is fine for a timing probe but means the pilot
cannot catch a long-sentence problem. **Fix:** the pilot samples across the
length distribution instead of taking the head.

## What was deliberately not changed

* **The model and the recipe.** 4.5M parameters, 104k steps, GAN at 8k, teacher
  forcing to zero. It produced an intelligible English voice from 19 hours of
  synthetic audio; there is no evidence a Spanish corpus needs a different
  recipe, and every deviation would be a guess.
* **`librosa.pyin`.** A faster tracker would cut `prepare` further, but it
  changes the pitch target and therefore the prosody. Parallelise, do not
  substitute.
* **The accent rules for `es-AR` and `es-CU`.** They are unvalidated by ear, and
  the run is `es-ES`. Changing them now would be churn.
* **The English front end.** Its fingerprint is pinned so released checkpoints
  keep loading; nothing in this fork touches it.
