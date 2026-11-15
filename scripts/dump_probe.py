import sys
sys.path.insert(0, "/root/probe" if len(sys.argv) > 1 else "/Users/maceo/blockvantage/tiny-voca/src")
sys.path.insert(0, "/Users/maceo/blockvantage/tiny-voca/scripts")
from espeak_probe import PROBES
from tiny_voca.text import phonemize
for text in PROBES:
    print(" ".join(phonemize(text).symbols))
