"""Upload the staged Spanish release to the Hugging Face Hub.

Two folders are staged under ``~/tiny-voca-artifacts/release-es-1/``:

    hf-model/     README.md (the model card), tiny-voca-es.pt/.onnx/.json,
                  samples/*.wav, SHA256SUMS
    hf-dataset/   README.md (the dataset card), LICENSE.md, data/<split>/*.flac
                  and metadata.jsonl, annotations/, voices/, text/,
                  generation/, SHA256SUMS

They go to two public repositories:

    model    https://huggingface.co/marcosmaceo/tiny-voca-es
    dataset  https://huggingface.co/datasets/marcosmaceo/tiny-voca-es

The script uploads whatever is staged, so replacing a staged file (for example
a fresh ONNX export) needs no change here. Re-run ``shasum -a 256`` into the
folder's SHA256SUMS and update the hashes in README.md when you do; the script
checks both.

    python scripts/release_hf.py              # dry run (the default): nothing is uploaded
    python scripts/release_hf.py --yes        # create the repos and upload

A dry run reads the staged files, verifies every SHA256SUMS entry (and that no
staged file is missing from it), checks that the hashes written in the model
card match the files, and prints what would be uploaded, with sizes and the
target URLs. It makes one read-only call to the Hub (whoami) to say which
account the saved token belongs to. With ``--yes`` the model folder goes up
with ``upload_folder`` and the dataset with ``upload_large_folder`` (resumable,
multi-commit; it keeps a ``.cache/`` folder inside the staged dataset folder,
which is not uploaded). Both repositories are created public with
``exist_ok=True``; existing files with the same path are overwritten, nothing is
deleted. Needs ``huggingface_hub`` and a login (``hf auth login``).
"""

import argparse
import hashlib
from pathlib import Path
import re
import sys

STAGING = Path("~/tiny-voca-artifacts/release-es-1").expanduser()
MODEL_REPO = "marcosmaceo/tiny-voca-es"
DATASET_REPO = "marcosmaceo/tiny-voca-es"
TARGETS = (
    # (label, repo type, repo id, folder, url)
    ("model", "model", MODEL_REPO, "hf-model", f"https://huggingface.co/{MODEL_REPO}"),
    ("dataset", "dataset", DATASET_REPO, "hf-dataset", f"https://huggingface.co/datasets/{DATASET_REPO}"),
)
SKIP = {".cache", ".DS_Store"}
CARD_HASH = re.compile(r"`([^`\s]+)`: `([0-9a-f]{64})`")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def staged_files(folder):
    """Every file that would be uploaded, as paths relative to the folder."""
    return sorted(path.relative_to(folder) for path in folder.rglob("*")
                  if path.is_file() and not (set(path.relative_to(folder).parts) & SKIP))


def size(count):
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1000 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1000


def verify(folder):
    """Check SHA256SUMS against the files. Returns a list of problems."""
    problems = []
    sums = folder / "SHA256SUMS"
    if not sums.is_file():
        return [f"{folder.name}: SHA256SUMS is missing"]
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip().lstrip("*")] = digest
    for name, digest in listed.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{folder.name}: listed but missing: {name}")
        elif sha256(path) != digest:
            problems.append(f"{folder.name}: hash differs: {name}")
    # The card and the sums file describe the others; every other file must be listed.
    for relative in staged_files(folder):
        if relative.as_posix() not in listed and relative.as_posix() not in {"README.md", "SHA256SUMS"}:
            problems.append(f"{folder.name}: staged but not in SHA256SUMS: {relative}")
    return problems


def verify_card(folder):
    """The model card names hashes as ``file``: ``sha256``; they must match the files."""
    problems = []
    card = folder / "README.md"
    if not card.is_file():
        return [f"{folder.name}: README.md is missing"]
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        problems.append(f"{folder.name}: README.md has no YAML front matter")
    for name, digest in CARD_HASH.findall(text):
        path = folder / name
        if path.is_file() and sha256(path) != digest:
            problems.append(f"{folder.name}: README.md gives a stale hash for {name}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Really create the repositories and upload (default: dry run)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded (the default; kept for clarity)")
    parser.add_argument("--staging", type=Path, default=STAGING, help="Folder holding hf-model/ and hf-dataset/")
    parser.add_argument("--only", choices=("model", "dataset"), help="Upload just one of the two")
    args = parser.parse_args()
    if args.yes and args.dry_run:
        parser.error("--yes and --dry-run exclude each other")
    staging = args.staging.expanduser()
    targets = [target for target in TARGETS if args.only in (None, target[0])]

    problems = []
    for label, kind, repo, name, url in targets:
        folder = staging / name
        if not folder.is_dir():
            problems.append(f"{folder} does not exist")
            continue
        files = staged_files(folder)
        total = sum((folder / path).stat().st_size for path in files)
        print(f"{label}: {folder}\n  -> {url}  ({kind} repo {repo}, public)\n  {len(files)} files, {size(total)}")
        bulk = {}
        for path in files:
            top = path.parts[0] if len(path.parts) > 1 else "."
            count, total_bytes = bulk.get(top, (0, 0))
            bulk[top] = (count + 1, total_bytes + (folder / path).stat().st_size)
        for top, (count, total_bytes) in sorted(bulk.items()):
            print(f"    {top + '/' if top != '.' else '(top level)':<14} {count:>6} files {size(total_bytes):>10}")
        if label == "model":
            for path in files:
                print(f"      {str(path):<28} {size((folder / path).stat().st_size):>10}")
        print("  verifying SHA256SUMS ...", flush=True)
        found = verify(folder) + verify_card(folder)
        print("  " + ("checksums and card hashes OK" if not found else f"{len(found)} problem(s)"))
        problems += found

    for problem in problems:
        print("PROBLEM:", problem, file=sys.stderr)

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder, upload_large_folder
        who = HfApi().whoami()["name"]
        print(f"Hub login: {who}")
        if who != MODEL_REPO.split("/")[0]:
            problems.append(f"logged in as {who}, repositories belong to {MODEL_REPO.split('/')[0]}")
    except Exception as error:  # a dry run does not need the Hub
        who = None
        print(f"Hub login: not available ({type(error).__name__}: {error})")
        if args.yes:
            problems.append("cannot reach the Hub with a saved token; run `hf auth login`")

    if not args.yes:
        print("\nDry run: nothing was uploaded. Run with --yes to upload.")
        return 1 if problems else 0
    if problems:
        print("Not uploading: fix the problems above first.", file=sys.stderr)
        return 1

    for label, kind, repo, name, url in targets:
        folder = staging / name
        create_repo(repo, repo_type=kind, exist_ok=True, private=False)
        print(f"uploading {label} to {url} ...", flush=True)
        if label == "model":
            upload_folder(repo_id=repo, repo_type=kind, folder_path=folder, ignore_patterns=[".cache/**", ".DS_Store"],
                          commit_message="Add tiny-voca-es (step 142,000, mixed excitation)")
        else:
            upload_large_folder(repo_id=repo, repo_type=kind, folder_path=folder,
                                ignore_patterns=[".cache/**", ".DS_Store"])
        print(f"done: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
