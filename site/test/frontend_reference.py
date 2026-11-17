"""Reference for site/test/frontend_golden.mjs: runs the Python front end.

PYTHONPATH=src TINY_VOCA_LANGUAGE=es .venv/bin/python site/test/frontend_reference.py in.json out.json

in.json is a list of strings; out.json gets, per string, either
{"symbols", "ids", "stress", "unknown"} or {"error": ExceptionClassName}.
"""
import json
import sys

from tiny_voca import text as T

texts = json.load(open(sys.argv[1], encoding="utf-8"))
out = []
for t in texts:
    try:
        p = T.phonemize(t)
        out.append({"symbols": list(p.symbols), "ids": list(p.ids), "stress": list(p.stress), "unknown": list(p.unknown)})
    except Exception as error:
        out.append({"error": type(error).__name__})
json.dump({"voice": T.ESPEAK_VOICE, "symbols": list(T.SYMBOLS), "results": out}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
