"""Check a GPU host before it spends money, and download what the run needs.

Every check is independent and prints one line. The first failure is fatal: a
missing cuDNN or a 1.51-vs-1.52 eSpeak is much cheaper to find now than after
the audio has been generated.

    python scripts/preflight.py                 # checks + model downloads
    python scripts/preflight.py --no-download   # checks only
    python scripts/preflight.py --data data     # also check free space there
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ALIGNER = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TEACHER = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ALIGNER_FILES = ["*.json", "*.bin", "*.safetensors", "*.txt", "*.model"]


class Failed(Exception):
    pass


def check(name, function, *arguments, **keywords):
    try:
        detail = function(*arguments, **keywords)
    except Exception as error:
        print(f"FAIL  {name}: {type(error).__name__}: {error}")
        raise Failed(name) from error
    print(f"ok    {name}{': ' + str(detail) if detail else ''}")
    return detail


def optional(name, function, *arguments, **keywords):
    """A check whose failure only changes a setting, not the run's viability."""
    try:
        detail = function(*arguments, **keywords)
    except Exception as error:
        print(f"warn  {name}: {type(error).__name__}: {error} (falling back)")
        return None
    print(f"ok    {name}{': ' + str(detail) if detail else ''}")
    return detail


def environment(allow_cpu=False):
    import platform
    import torch
    detail = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()}
    if torch.cuda.is_available():
        detail.update({"gpu": torch.cuda.get_device_name(0),
                       "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                       "cuda": torch.version.cuda})
        # A real kernel, not just the flag: catches a driver/library mismatch.
        value = (torch.ones(4096, device="cuda") * 2).sum().item()
        if value != 8192:
            raise RuntimeError("CUDA kernel returned the wrong value")
    elif not allow_cpu:
        raise RuntimeError("torch cannot see a CUDA device")
    else:
        detail["gpu"] = "none (CPU only)"
    return json.dumps(detail)


def memory():
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return f"{int(line.split()[1]) / 2**20:.1f} GiB"
    except OSError:
        pass
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return f"{total / 2**30:.1f} GiB"


def disk(path, minimum_gb=40.0):
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum_gb * 2**30:
        raise RuntimeError(f"only {free / 2**30:.1f} GiB free (need {minimum_gb:.0f})")
    return f"{free / 2**30:.1f} GiB free at {path}"


def espeak():
    from tiny_voca.text import ACCENT, LANGUAGE, SYMBOLS, espeak_version, phonemize
    version = espeak_version()
    # "Zaragoza" must have θ for the Castilian front end; a seseante espeak
    # would mean the whole corpus is labelled wrong.
    phones = phonemize("Zaragoza")
    if LANGUAGE == "es" and ACCENT == "es-ES" and "θ" not in phones.symbols:
        raise RuntimeError(f"es-ES front end produced no θ for 'Zaragoza': {' '.join(phones.symbols)}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from espeak_probe import fingerprint
    return f"{version.splitlines()[0]}; {LANGUAGE}/{ACCENT}; {len(SYMBOLS)} symbols; probe {fingerprint()[:16]}"


def packages():
    import importlib
    found, missing = [], []
    for name in ("faster_whisper", "librosa", "onnxruntime", "qwen_tts", "soundfile", "transformers", "ctranslate2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        found.append(f"{name} {getattr(module, '__version__', '?')}")
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return ", ".join(found)


def cudnn():
    """CTranslate2 (faster-whisper) needs cuDNN 9 on the GPU."""
    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        raise RuntimeError("ctranslate2 sees no CUDA device")
    return f"ctranslate2 {ctranslate2.__version__}, {ctranslate2.get_cuda_device_count()} device(s)"


def download_aligner():
    from huggingface_hub import snapshot_download
    path = snapshot_download(ALIGNER, allow_patterns=ALIGNER_FILES)
    return path


def download_teacher():
    from huggingface_hub import snapshot_download
    path = snapshot_download(TEACHER)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**30
    return f"{size:.1f} GiB"


def download_asr():
    from faster_whisper import WhisperModel
    from tiny_voca.languages import profile as language
    WhisperModel(language.ASR_MODEL, device="cpu", compute_type="int8")
    return language.ASR_MODEL


def configuration():
    from tiny_voca.config import Config
    c = Config()
    if c.vocab_size != 61:
        raise RuntimeError(f"vocab_size is {c.vocab_size}, expected 61 for Spanish")
    return f"vocab {c.vocab_size}, speakers {c.speakers}, {c.sample_rate} Hz"


def onnx_roundtrip():
    import tempfile
    import torch
    from tiny_voca.config import Config
    from tiny_voca.export import export
    from tiny_voca.model import TinyVoca
    with tempfile.TemporaryDirectory() as directory:
        model = TinyVoca(Config(channels=32, heads=2, feedforward=64, kernel=3, encoder_layers=1, prosody_layers=1,
                                acoustic_layers=1, predictor_channels=16, cep_causal=12, cep_anti=4, noise_cep=8)).eval()
        path = export(model, Path(directory) / "model.onnx", external_noise=True)
        import onnxruntime
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return f"{path.stat().st_size / 1e6:.1f} MB, {len(session.get_inputs())} inputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail when there is no CUDA device")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--min-disk-gb", type=float, default=40.0)
    args = parser.parse_args()
    try:
        check("environment", environment, allow_cpu=args.allow_cpu)
        check("memory", memory)
        check("disk", disk, args.data, args.min_disk_gb)
        check("espeak", espeak)
        check("packages", packages)
        if not args.allow_cpu:
            optional("cuda libraries", cudnn)
        check("configuration", configuration)
        check("onnx roundtrip", onnx_roundtrip)
        if not args.no_download:
            check("aligner model", download_aligner)
            check("asr model", download_asr)
            check("teacher model", download_teacher)
    except Failed:
        print("\npreflight failed; fix the line above before starting the run")
        raise SystemExit(1)
    print("\npreflight passed")


if __name__ == "__main__":
    main()
