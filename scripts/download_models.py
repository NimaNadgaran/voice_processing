"""Pre-download pretrained weights so the first real run is instant (or offline).

    python scripts/download_models.py --list
    python scripts/download_models.py --all
    python scripts/download_models.py deepfilternet sepformer

Only backends whose python package is already installed are touched -- this
script never installs packages and never starts training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.utils import PRETRAINED_DIR, human_time, module_available  # noqa: E402


# --------------------------------------------------------------------------- #
def dl_deepfilternet() -> str:
    # The pip package only installs on Python <=3.11; everywhere else the method
    # runs the official standalone binary instead, so fetch whichever applies.
    if module_available("df") and module_available("torch"):
        from df.enhance import init_df  # type: ignore

        init_df(config_allow_defaults=True)
        return "DeepFilterNet3 ready (pip package)"

    from src.denoising.methods.deepfilternet import download_deep_filter

    return "DeepFilterNet3 ready (standalone binary: %s)" % download_deep_filter()


def dl_denoiser() -> str:
    from denoiser import pretrained  # type: ignore

    pretrained.dns64()
    return "Demucs denoiser (dns64) ready"


def dl_demucs() -> str:
    from demucs.pretrained import get_model  # type: ignore

    get_model("htdemucs")
    return "htdemucs ready"


def dl_ecapa() -> str:
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
    except Exception:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(PRETRAINED_DIR / "ecapa"),
    )
    return "ECAPA-TDNN speaker embeddings ready (improves speaker counting a lot)"


def dl_sepformer() -> str:
    try:
        from speechbrain.inference.separation import SepformerSeparation  # type: ignore
    except Exception:
        from speechbrain.pretrained import SepformerSeparation  # type: ignore

    out = []
    for model_id in ("speechbrain/sepformer-whamr16k", "speechbrain/sepformer-wsj03mix"):
        SepformerSeparation.from_hparams(
            source=model_id, savedir=str(PRETRAINED_DIR / model_id.replace("/", "__"))
        )
        out.append(model_id)
    return "SepFormer ready: " + ", ".join(out)


def dl_asteroid() -> str:
    from asteroid.models import BaseModel  # type: ignore

    for model_id in ("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
                     "JorisCos/ConvTasNet_Libri3Mix_sepclean_16k"):
        BaseModel.from_pretrained(model_id)
    return "Asteroid Conv-TasNet (2 + 3 src) ready"


def dl_pyannote() -> str:
    import os

    from pyannote.audio import Pipeline  # type: ignore

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("set HF_TOKEN and accept the model licences (see models/README.md)")
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    return "pyannote 3.1 pipeline ready"


def dl_clearvoice() -> str:
    from clearvoice import ClearVoice  # type: ignore

    ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    return "MossFormer2 ready"


def dl_nemo() -> str:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer  # type: ignore

    NeuralDiarizer.from_pretrained(model_name="diar_msdd_telephonic")
    return "NeMo MSDD ready"


#  A module of None means "no package needed" -- the target fetches something
#  self-contained and is always available.
TARGETS = {
    #  name            module to probe        size hint   downloader
    "deepfilternet": (None, "~27 MB", dl_deepfilternet),
    "demucs_denoiser": ("denoiser", "~130 MB", dl_denoiser),
    "demucs_vocals": ("demucs", "~80 MB", dl_demucs),
    "ecapa": ("speechbrain", "~80 MB", dl_ecapa),
    "sepformer": ("speechbrain", "~220 MB", dl_sepformer),
    "asteroid": ("asteroid", "~40 MB", dl_asteroid),
    "pyannote": ("pyannote.audio", "~30 MB", dl_pyannote),
    "clearvoice": ("clearvoice", "~200 MB", dl_clearvoice),
    "nemo": ("nemo", "~90 MB", dl_nemo),
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="names to fetch (default: none)")
    ap.add_argument("--all", action="store_true", help="every target whose package is installed")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if args.list or (not args.targets and not args.all):
        print("%-18s %-16s %-10s %s" % ("target", "package", "size", "status"))
        print("-" * 66)
        for name, (module, size, _) in TARGETS.items():
            installed = module is None or module_available(module)
            print("%-18s %-16s %-10s %s"
                  % (name, module or "-", size,
                     "ready" if installed else "package missing"))
        print("\nfetch with:  python scripts/download_models.py --all")
        print("         or:  python scripts/download_models.py deepfilternet sepformer")
        return 0

    names = sorted(TARGETS) if args.all else args.targets
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        print("unknown target(s): %s\nvalid: %s" % (", ".join(unknown), ", ".join(sorted(TARGETS))))
        return 2

    failures = 0
    for name in names:
        module, size, fetch = TARGETS[name]
        if module is not None and not module_available(module):
            print("[skip] %-18s package '%s' is not installed" % (name, module))
            continue
        print("[ .. ] %-18s downloading %s ..." % (name, size), flush=True)
        started = time.time()
        try:
            message = fetch()
            print("[ ok ] %-18s %s  (%s)" % (name, message, human_time(time.time() - started)))
        except Exception as exc:
            failures += 1
            print("[fail] %-18s %s: %s" % (name, type(exc).__name__, exc))

    print("\ncache: %s" % PRETRAINED_DIR)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
