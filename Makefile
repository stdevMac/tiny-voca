# tiny-voca-es: a Spanish fork of tiny-voca. See docs/spanish.md and docs/training-es.md.
#
#   make venv test                 set up and verify on the laptop
#   make pilot                     whole chain on macOS voices, no GPU (~1 h)
#   --- on a GPU host ---
#   make scripts                   Spanish sentence list (CC0 + templates)
#   make voices                    design F2/M2 and write their reference clips
#   make audio                     teacher TTS -> data/audio/manifest.jsonl
#   make features                  acoustic features
#   make annotate                  phonemes + durations (forced alignment)
#   make train                     the model
#   make evaluate export           scores and the ONNX file
#   make speak TEXT="Hola"         listen on the laptop
#
# Every stage writes into its own directory and refuses to overwrite a finished
# one, so a rerun is a deliberate act: delete the directory or pass a new name.

PY      ?= .venv/bin/python
DATA    ?= data
RUN     ?= es-1
STEPS   ?= 104000
VOICES  ?= voices/es-ES.json
LIMIT   ?=
DEVICE  ?= cuda
TEXT    ?= Hola, ¿cómo estás? Esto es tiny voca en español.
OUT     ?= out.wav
VOICE   ?= F2
CKPT    ?= $(DATA)/runs/$(RUN)/inference.pt

.PHONY: help venv test pilot scripts voices audio features annotate train evaluate export speak clean

help:
	@sed -n '2,15p' Makefile | sed 's/^# \{0,1\}//'

venv:
	uv venv --python 3.13 .venv
	uv pip install --python $(PY) -e ".[prepare,annotate,evaluate,export]"

test:
	PYTHONPATH=src:tests $(PY) -m unittest discover -s tests

# --- laptop only: a stand-in teacher, to prove the chain before paying for a GPU ---
pilot:
	$(PY) scripts/build_text_es.py --collector $(DATA)/common-voice/sentence-collector.txt \
	  --wiki $(DATA)/common-voice/wiki.es.txt --output $(DATA)/scripts-pilot.jsonl --count 300 --validation 20 --test 20
	$(PY) scripts/generate_audio_local.py --scripts $(DATA)/scripts-pilot.jsonl --root $(DATA)/pilot \
	  --voices "Mónica|Grandpa (Spanish (Spain))" --limit 120
	$(PY) scripts/split_manifest.py --root $(DATA)/pilot
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/pilot/train.jsonl --split train --output $(DATA)/pilot/features
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/pilot/validation.jsonl --split validation --output $(DATA)/pilot/features
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/pilot/test.jsonl --split test --output $(DATA)/pilot/features
	$(PY) -m tiny_voca.annotate --features $(DATA)/pilot/features --output $(DATA)/pilot/annotations \
	  --dataset-revision local-pilot --device cpu
	$(PY) -m tiny_voca.train --train $(DATA)/pilot/annotations/train.jsonl \
	  --validation $(DATA)/pilot/annotations/validation.jsonl --output $(DATA)/pilot/runs/pilot \
	  --device cpu --steps 200 --batch-frames 4000 --max-batch 16 --save-every 100 --log-every 50 \
	  --gan-start 100 --validation-size 8
	@echo "pilot done: $(DATA)/pilot/runs/pilot/inference.pt"

# --- GPU host ---
scripts:
	$(PY) scripts/build_text_es.py --collector $(DATA)/common-voice/sentence-collector.txt \
	  --wiki $(DATA)/common-voice/wiki.es.txt --output $(DATA)/scripts-es.jsonl --count 9000

voices:
	$(PY) scripts/design_voices.py --root $(DATA)/voices-es --config $(VOICES) --device $(DEVICE)

audio:
	$(PY) scripts/generate_audio.py --scripts $(DATA)/scripts-es.jsonl --root $(DATA)/audio-es \
	  --references $(DATA)/voices-es/references --voices $(DATA)/voices-es/voices.json --deadline-file $(DATA)/budget.json

features:
	$(PY) scripts/split_manifest.py --root $(DATA)/audio-es
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/audio-es/train.jsonl --split train --output $(DATA)/features-es
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/audio-es/validation.jsonl --split validation --output $(DATA)/features-es
	$(PY) -m tiny_voca.prepare --inputs $(DATA)/audio-es/test.jsonl --split test --output $(DATA)/features-es

annotate:
	$(PY) -m tiny_voca.annotate --features $(DATA)/features-es --output $(DATA)/annotations-es \
	  --dataset-revision $(RUN) --device $(DEVICE)

train:
	$(PY) -m tiny_voca.train --train $(DATA)/annotations-es/train.jsonl \
	  --validation $(DATA)/annotations-es/validation.jsonl --output $(DATA)/runs/$(RUN) \
	  --device $(DEVICE) --steps $(STEPS) --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
	  --deadline-file $(DATA)/budget.json

evaluate:
	$(PY) -m tiny_voca.evaluate --checkpoint $(DATA)/runs/$(RUN)/inference.pt \
	  --annotations $(DATA)/annotations-es --output $(DATA)/eval/$(RUN) --tag $(RUN) --device $(DEVICE)

export:
	$(PY) -m tiny_voca.export --checkpoint $(DATA)/runs/$(RUN)/inference.pt --output $(DATA)/runs/$(RUN)/tiny-voca-es.onnx

speak:
	$(PY) -m tiny_voca.infer --checkpoint $(CKPT) --voice $(VOICE) --text "$(TEXT)" --output $(OUT)

clean:
	rm -rf $(DATA)/pilot
