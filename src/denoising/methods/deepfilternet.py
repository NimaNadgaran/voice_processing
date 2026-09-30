"""Method 3 -- DeepFilterNet 3 (the headline neural denoiser).

Two ways to run it
------------------
1. **The PyTorch package** -- ``pip install torch torchaudio`` then
   ``pip install deepfilternet``.  This is the reference implementation, but it
   only installs on **Python 3.8-3.11**: its native extension ``DeepFilterLib``
   publishes wheels up to cp311 only, the sdist is built with ``pyo3 0.19``
   (which caps at CPython 3.12), and ``deepfilternet`` pins ``numpy<2``.

2. **The standalone ``deep-filter`` binary** -- a single executable published on
   the project's GitHub releases with the DFN3 weights compiled in, running the
   ONNX graph through ``tract``.  No Python, no torch, no numpy pin, no Rust
   toolchain.  This is what makes the method work on 3.12+::

       python scripts/download_models.py deepfilternet

   It lands in ``models/pretrained/deepfilternet/``.  Set ``DEEP_FILTER_BIN`` to
   point somewhere else, or just put ``deep-filter`` on your PATH.

Whichever is present is used; the package wins when both are, since it avoids a
round trip through disk.  Same model, same weights, same output either way.

Why it is good
--------------
DeepFilterNet works at 48 kHz and predicts *deep filters* (complex multi-frame
filters) for the low band plus an ERB gain envelope for the high band.  That two
stage design is why it removes a lot of noise while keeping the voice crisp, at
roughly 0.05x real time on a single CPU core.

Reference: Schroeter et al., "DeepFilterNet2/DeepFilterNet3", INTERSPEECH 2022 /
ICASSP 2023.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.audio_io import load_audio, save_audio
from ...core.errors import AppError
from ...core.registry import register_denoiser
from ...core.types import AudioBuffer, MethodInfo
from ...core.utils import PRETRAINED_DIR, env_flag, module_available
from ..base import BaseDenoiser

# --------------------------------------------------------------------------- #
#  the standalone binary
# --------------------------------------------------------------------------- #
DEEP_FILTER_VERSION = "0.5.6"
DEEP_FILTER_DIR = PRETRAINED_DIR / "deepfilternet"

#  (platform.system(), platform.machine()) -> release asset name
_ASSETS = {
    ("Windows", "AMD64"): "deep-filter-%s-x86_64-pc-windows-msvc.exe",
    ("Linux", "x86_64"): "deep-filter-%s-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "deep-filter-%s-aarch64-unknown-linux-gnu",
    ("Linux", "armv7l"): "deep-filter-%s-armv7-unknown-linux-gnueabihf",
    ("Darwin", "x86_64"): "deep-filter-%s-x86_64-apple-darwin",
    ("Darwin", "arm64"): "deep-filter-%s-aarch64-apple-darwin",
}
_RELEASE_URL = "https://github.com/Rikorose/DeepFilterNet/releases/download/v%s/%s"

DOWNLOAD_HINT = "python scripts/download_models.py deepfilternet"


def deep_filter_asset() -> Optional[str]:
    """Release asset for this machine, or None if there is no build for it."""
    template = _ASSETS.get((platform.system(), platform.machine()))
    return template % DEEP_FILTER_VERSION if template else None


def deep_filter_url() -> Optional[str]:
    asset = deep_filter_asset()
    return _RELEASE_URL % (DEEP_FILTER_VERSION, asset) if asset else None


def deep_filter_path() -> Path:
    """Where :func:`download_deep_filter` puts the binary."""
    name = "deep-filter.exe" if os.name == "nt" else "deep-filter"
    return DEEP_FILTER_DIR / name


def deep_filter_bin() -> Optional[Path]:
    """Find a usable ``deep-filter``: env var, then bundled copy, then PATH."""
    override = os.environ.get("DEEP_FILTER_BIN", "").strip()
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None

    bundled = deep_filter_path()
    if bundled.is_file():
        return bundled

    found = shutil.which("deep-filter")
    return Path(found) if found else None


def download_deep_filter(force: bool = False) -> Path:
    """Fetch the release binary into ``models/pretrained/deepfilternet/``."""
    import urllib.request

    target = deep_filter_path()
    if target.is_file() and not force:
        return target

    url = deep_filter_url()
    if not url:
        raise AppError(
            "no prebuilt deep-filter binary for %s/%s"
            % (platform.system(), platform.machine()),
            fix="build it from source: cargo install --git "
            "https://github.com/Rikorose/DeepFilterNet deep_filter "
            "--features bin,tract,default-model",
            kind="unsupported-platform",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(target)
    if os.name != "nt":
        target.chmod(0o755)
    return target


# --------------------------------------------------------------------------- #
@register_denoiser
class DeepFilterNetDenoiser(BaseDenoiser):
    info = MethodInfo(
        key="deepfilternet",
        name="DeepFilterNet 3",
        kind="denoise",
        family="deep-pretrained",
        description=(
            "State-of-the-art 48 kHz real-time neural denoiser (deep filtering + ERB "
            "gains). Removes fans, keyboards, street noise and reverb tails while "
            "keeping speech natural."
        ),
        speed="fast",
        quality=5,
        needs_gpu=False,
        offline=True,  # after the binary / weights are fetched once
        pip=[],  # neither backend is a plain pip dependency -- see check_available
        install_hint=DOWNLOAD_HINT,
        notes=(
            "Runs at 48 kHz internally. Uses the pip package on Python <=3.11, "
            "otherwise the official standalone deep-filter binary (~27 MB)."
        ),
    )
    target_sr = 48000
    restore_sr = True

    #  `deepfilternet` + `DeepFilterLib` cannot be installed above this.
    MAX_PYTHON = (3, 11)

    # ------------------------------------------------------------------ #
    #  availability
    # ------------------------------------------------------------------ #
    def _torch_backend_ready(self) -> bool:
        return module_available("torch") and module_available("df")

    def check_available(self) -> Tuple[bool, str]:
        if self._torch_backend_ready():
            return True, ""
        if deep_filter_bin() is not None:
            return True, ""

        if deep_filter_asset() is None:
            return False, (
                "no deep-filter build for %s/%s, and the pip package is not installed"
                % (platform.system(), platform.machine())
            )
        if sys.version_info[:2] > self.MAX_PYTHON:
            return False, (
                "the deep-filter binary has not been downloaded yet (the pip "
                "package cannot be installed on Python %d.%d)" % sys.version_info[:2]
            )
        return False, "missing the deep-filter binary and the 'deepfilternet' package"

    def describe(self) -> MethodInfo:
        info = super().describe()
        if not info.available:
            info.install_hint = DOWNLOAD_HINT
        return info

    # ------------------------------------------------------------------ #
    #  lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._torch_backend_ready():
            os.environ.setdefault("DF_LOG_LEVEL", "ERROR")
            from df.enhance import init_df  # type: ignore

            model, df_state, _ = init_df(config_allow_defaults=True)
            self._model = ("torch", (model, df_state))
        else:
            exe = deep_filter_bin()
            if exe is None:  # check_available() already ran, so this is a race
                raise AppError(
                    "the deep-filter binary disappeared", fix=DOWNLOAD_HINT,
                    kind="missing-dependency",
                )
            self._model = ("bin", exe)
        self._loaded = True

    # ------------------------------------------------------------------ #
    #  inference
    # ------------------------------------------------------------------ #
    def _denoise(self, audio: AudioBuffer):
        backend, handle = self._model
        if backend == "torch":
            return self._denoise_torch(audio, handle)
        return self._denoise_bin(audio, handle)

    def _denoise_torch(self, audio: AudioBuffer, handle: Any):
        import torch  # type: ignore
        from df.enhance import enhance  # type: ignore

        model, df_state = handle
        tensor = torch.from_numpy(np.ascontiguousarray(audio.samples)).float().unsqueeze(0)
        with torch.no_grad():
            out = enhance(model, df_state, tensor, pad=True)
        arr = out.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return AudioBuffer(arr, audio.sr), {
            "backend": "deepfilternet3",
            "internal_sr": self.target_sr,
            "model_params_m": 2.3,
        }

    def _denoise_bin(self, audio: AudioBuffer, exe: Path):
        """Round trip through a temp wav -- the binary is a file-in/file-out CLI."""
        with tempfile.TemporaryDirectory(prefix="deepfilter_") as tmpdir:
            tmp = Path(tmpdir)
            in_wav = tmp / "input.wav"
            out_dir = tmp / "out"
            save_audio(in_wav, audio, subtype="PCM_16")

            cmd = [
                str(exe),
                "--compensate-delay",  # align output with input for the metrics
                "--output-dir", str(out_dir),
            ]
            atten = os.environ.get("DEEP_FILTER_ATTEN_LIM_DB", "").strip()
            if atten:
                cmd += ["--atten-lim-db", atten]
            if env_flag("DEEP_FILTER_POST_FILTER"):
                cmd.append("--pf")
            cmd.append(str(in_wav))

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            out_wav = out_dir / in_wav.name
            if proc.returncode != 0 or not out_wav.is_file():
                raise AppError(
                    "deep-filter failed: %s" % (_last_line(proc.stdout) or
                                                "exit code %d" % proc.returncode),
                    fix="re-download it with: " + DOWNLOAD_HINT,
                    kind="backend-failed",
                )
            enhanced = load_audio(out_wav)

        metrics: Dict[str, Any] = {
            "backend": "deepfilternet3-bin",
            "internal_sr": self.target_sr,
            "model_params_m": 2.3,
            "binary": str(exe),
        }
        return AudioBuffer(enhanced.samples.astype(np.float32), enhanced.sr), metrics


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""
