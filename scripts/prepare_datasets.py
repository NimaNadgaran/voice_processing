"""Check / organise training corpora. Never downloads multi-GB data on its own.

    python scripts/prepare_datasets.py --check
    python scripts/prepare_datasets.py --links
    python scripts/prepare_datasets.py --organize-librispeech D:/LibriSpeech/train-clean-100
    python scripts/prepare_datasets.py --flat-to-speakers D:/my_recordings

``--organize-librispeech`` and ``--flat-to-speakers`` create *symlinks* where the
OS allows it and fall back to copying otherwise, so nothing is duplicated
needlessly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.utils import DATASET_DIR, human_size  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")

LINKS = """
CLEAN SPEECH
  LibriSpeech train-clean-100 (6 GB, 251 speakers, CC BY 4.0)
      https://www.openslr.org/resources/12/train-clean-100.tar.gz
  LibriSpeech train-clean-360 (23 GB, 921 speakers)
      https://www.openslr.org/resources/12/train-clean-360.tar.gz
  VCTK / VoiceBank (11 GB, 110 speakers, ODC-By)
      https://datashare.ed.ac.uk/handle/10283/3443
  Common Voice (any language, CC0)
      https://commonvoice.mozilla.org/en/datasets

NOISE
  DEMAND (4 GB, 18 environments, CC BY-SA)
      https://zenodo.org/records/1227121
  MUSAN (11 GB, noise + music + babble, CC BY 4.0)
      https://www.openslr.org/resources/17/musan.tar.gz
  ESC-50 (600 MB, small and quick, CC BY-NC)
      https://github.com/karolpiczak/ESC-50/archive/master.zip
  FSDnoisy18k (10 GB)
      https://zenodo.org/records/2529934

SPEAKERS (for the separator -- one folder per speaker)
  VoxCeleb1 (39 GB, 1251 speakers, CC BY 4.0)
      https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
  LibriSpeech works directly: --organize-librispeech <path>

Unpack anywhere, then point the training config at it, or organise into:
  data/datasets/clean/      data/datasets/noise/      data/datasets/speakers/
"""


# --------------------------------------------------------------------------- #
def scan(root: Path):
    files, total = [], 0
    for ext in AUDIO_EXTS:
        for f in root.rglob("*" + ext):
            if f.is_file():
                files.append(f)
                total += f.stat().st_size
    return files, total


def cmd_check() -> int:
    print("dataset root: %s\n" % DATASET_DIR)
    any_found = False
    for name, purpose in (("clean", "denoiser: clean speech"),
                          ("noise", "denoiser: noise (optional)"),
                          ("speakers", "separator: one folder per speaker")):
        folder = DATASET_DIR / name
        if not folder.exists():
            print("  %-10s MISSING     %s" % (name, purpose))
            continue
        files, total = scan(folder)
        any_found = any_found or bool(files)
        extra = ""
        if name == "speakers":
            groups = defaultdict(int)
            for f in files:
                try:
                    groups[f.relative_to(folder).parts[0]] += 1
                except (ValueError, IndexError):
                    groups[f.stem] += 1
            extra = "  %d speaker folder(s)" % len(groups)
            if len(groups) < 20:
                extra += "   ! fewer than 20 speakers: the separator will overfit"
        print("  %-10s %5d files  %9s  %s%s" % (name, len(files), human_size(total), purpose, extra))

    if not any_found:
        print("\nnothing found. run with --links for download instructions.")
    else:
        print("\nrough rule: ~1 GB of 16 kHz mono wav ~ 9 hours of audio.")
    return 0


def cmd_links() -> int:
    print(LINKS)
    return 0


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def cmd_organize_librispeech(source: Path, out: Path) -> int:
    """LibriSpeech <spk>/<chapter>/*.flac  ->  out/<spk>/*.flac"""
    if not source.exists():
        print("no such folder: %s" % source)
        return 2
    count, speakers = 0, set()
    for speaker_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        for f in speaker_dir.rglob("*"):
            if f.suffix.lower() in AUDIO_EXTS and f.is_file():
                _link_or_copy(f, out / speaker_dir.name / f.name)
                count += 1
                speakers.add(speaker_dir.name)
    print("organised %d files from %d speakers -> %s" % (count, len(speakers), out))
    if len(speakers) < 20:
        print("! fewer than 20 speakers -- the separator will memorise voices instead of separating")
    return 0


def cmd_flat_to_speakers(source: Path, out: Path) -> int:
    """A flat folder -> out/<filestem>/<file>, i.e. one 'speaker' per file."""
    if not source.exists():
        print("no such folder: %s" % source)
        return 2
    count = 0
    for f in sorted(source.rglob("*")):
        if f.suffix.lower() in AUDIO_EXTS and f.is_file():
            _link_or_copy(f, out / f.stem / f.name)
            count += 1
    print("organised %d files -> %s (one 'speaker' per file)" % (count, out))
    print("! this is a weak fallback: real speaker labels are much better.")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--organize-librispeech", metavar="PATH")
    ap.add_argument("--flat-to-speakers", metavar="PATH")
    ap.add_argument("--out", default=str(DATASET_DIR / "speakers"))
    args = ap.parse_args()

    if args.links:
        return cmd_links()
    if args.organize_librispeech:
        return cmd_organize_librispeech(Path(args.organize_librispeech), Path(args.out))
    if args.flat_to_speakers:
        return cmd_flat_to_speakers(Path(args.flat_to_speakers), Path(args.out))
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
