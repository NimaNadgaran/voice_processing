"""Audio loading / saving / resampling with graceful backend fallbacks.

Priority order
--------------
load   : soundfile  ->  librosa/audioread  ->  ffmpeg subprocess  ->  wave stdlib
save   : soundfile  ->  wave stdlib (16-bit PCM)
resample: soxr      ->  scipy.signal.resample_poly  ->  librosa  ->  linear interp

**Video files** (mp4, mov, mkv, avi, ...) are supported: the audio track is
extracted with ffmpeg and everything downstream sees ordinary mono audio. The
video itself is never decoded -- we pass ``-vn`` so only the audio stream is
touched, which keeps a 2 GB screen recording cheap to read.

ffmpeg is located in this order:
1. a system ``ffmpeg`` on PATH,
2. the binary bundled by the ``imageio-ffmpeg`` wheel (``pip install
   imageio-ffmpeg`` -- no system install, works on Windows without admin),
3. ``FFMPEG_BINARY`` if you want to point at a specific build.

Nothing here imports torch, so the app boots on a bare `pip install -r
requirements.txt`.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .errors import UnreadableAudio
from .types import AudioBuffer
from .utils import module_available

#: containers that hold audio only
AUDIO_EXTS = {
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac",
    ".wma", ".aiff", ".aif", ".aifc", ".caf", ".amr", ".au", ".w64", ".wv",
}

#: video containers -- we pull the audio track out of these
VIDEO_EXTS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".3g2", ".ogv", ".asf", ".rm",
}

SUPPORTED_EXTS = AUDIO_EXTS | VIDEO_EXTS


def is_video(path: "str | Path") -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


# --------------------------------------------------------------------------- #
#  ffmpeg discovery
# --------------------------------------------------------------------------- #
_FFMPEG_CACHE: Dict[str, Optional[str]] = {}


def ffmpeg_exe() -> Optional[str]:
    """Path to a usable ffmpeg, or None. Result is cached."""
    if "exe" in _FFMPEG_CACHE:
        return _FFMPEG_CACHE["exe"]

    found: Optional[str] = None
    explicit = os.environ.get("FFMPEG_BINARY", "").strip()
    if explicit and Path(explicit).exists():
        found = explicit
    if not found:
        found = shutil.which("ffmpeg")
    if not found and module_available("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg  # type: ignore

            candidate = imageio_ffmpeg.get_ffmpeg_exe()
            if candidate and Path(candidate).exists():
                found = candidate
        except Exception:
            found = None

    _FFMPEG_CACHE["exe"] = found
    return found


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


FFMPEG_INSTALL_HINT = (
    "pip install imageio-ffmpeg  (bundles ffmpeg, no admin rights needed) "
    "-- or install ffmpeg system-wide and put it on PATH"
)


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
def load_audio(
    path: "str | Path",
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> AudioBuffer:
    """Read any audio *or video* file into a mono float32 :class:`AudioBuffer`.

    For a video, the audio track is extracted and the picture is ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("audio file not found: %s" % path)

    video = is_video(path)
    if video and not ffmpeg_available():
        raise UnreadableAudio(
            "%s is a video file, and reading the audio out of a video needs ffmpeg."
            % path.name,
            fix=FFMPEG_INSTALL_HINT,
        )

    # A video container goes straight to ffmpeg: soundfile/librosa would only
    # waste time failing on it.
    readers = (_read_ffmpeg,) if video else (_read_soundfile, _read_librosa, _read_ffmpeg, _read_wave)

    last_error: Optional[Exception] = None
    for reader in readers:
        try:
            data, sr = reader(path)
        except Exception as exc:  # try the next backend
            last_error = exc
            continue
        if data is None:
            continue
        if data.size == 0:
            last_error = UnreadableAudio("the decoded track is empty")
            continue
        buf = AudioBuffer(data, sr)
        if target_sr and buf.sr != target_sr:
            buf = resample(buf, target_sr)
        return buf

    raise UnreadableAudio(*_decode_failure(path, video, last_error))


def _read_soundfile(path: Path) -> Tuple[Optional[np.ndarray], int]:
    if not module_available("soundfile"):
        return None, 0
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data, int(sr)


def _read_librosa(path: Path) -> Tuple[Optional[np.ndarray], int]:
    if not module_available("librosa"):
        return None, 0
    import librosa

    data, sr = librosa.load(str(path), sr=None, mono=True)
    return np.asarray(data, dtype=np.float32), int(sr)


def _read_ffmpeg(path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Decode through ffmpeg into a temp wav.

    Handles m4a/webm/anything, and pulls the audio track out of video files:
    ``-vn`` drops the picture and ``-map 0:a:0`` takes the first audio stream,
    so a 2 GB screen recording costs about as much as its audio alone.
    """
    exe = ffmpeg_exe()
    if not exe:
        return None, 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(path),
            "-vn",              # ignore the video stream entirely
            "-map", "0:a:0",    # first audio track
            "-ac", "1",         # downmix to mono
            "-c:a", "pcm_s16le",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        if proc.returncode != 0 or not out.exists():
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            # ffmpeg words this differently across versions, so rather than
            # pattern-matching its prose we just ask what streams the file has.
            if not has_audio_stream(path):
                raise UnreadableAudio(
                    "%s has no audio track -- there is nothing to denoise." % path.name,
                    fix="Check the file actually plays with sound, then export it again with audio included.",
                )
            raise UnreadableAudio(
                "ffmpeg could not decode %s." % path.name,
                fix=_first_useful_line(stderr) or "The file may be corrupt or use an unsupported codec.",
            )
        return _read_wave(out)


def has_audio_stream(path: "str | Path") -> bool:
    """True if ffmpeg can see at least one audio stream in the file.

    ``ffmpeg -i FILE`` with no output prints the stream table and exits
    non-zero; that is the cheapest probe that works with the imageio-ffmpeg
    binary, which does not ship ffprobe.
    """
    exe = ffmpeg_exe()
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-i", str(path)],
            capture_output=True, timeout=120,
        )
    except Exception:
        return False
    info = (proc.stderr or b"").decode("utf-8", "replace")
    return any("Audio:" in line for line in info.splitlines() if "Stream #" in line)


def _first_useful_line(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line and not line.startswith("["):
            return line[:200]
    return ""


def _decode_failure(path: Path, video: bool, last_error: Optional[Exception]) -> Tuple[str, str]:
    """Turn 'every reader failed' into a sentence plus an actionable fix."""
    if isinstance(last_error, UnreadableAudio):
        return last_error.message, last_error.fix
    if video:
        return ("Could not read the audio track out of %s." % path.name, FFMPEG_INSTALL_HINT)
    if not ffmpeg_available():
        return (
            "Could not decode %s -- the format is not one the built-in reader understands." % path.name,
            FFMPEG_INSTALL_HINT + " (that adds mp3, m4a, video and everything else)",
        )
    return (
        "Could not decode %s. The file may be corrupt or truncated." % path.name,
        "Try re-exporting it as WAV or MP3.",
    )


def _read_wave(path: Path) -> Tuple[Optional[np.ndarray], int]:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError("unsupported PCM width: %d bytes" % width)
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(np.iinfo(dtype).max)
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data.astype(np.float32), int(sr)


def load_from_bytes(raw: bytes, filename: str = "upload.wav", target_sr: Optional[int] = None) -> AudioBuffer:
    """Decode an in-memory upload (writes to a temp file for the ffmpeg path)."""
    if module_available("soundfile") and not is_video(filename):
        try:
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            buf = AudioBuffer(data, int(sr))
            return resample(buf, target_sr) if target_sr and buf.sr != target_sr else buf
        except Exception:
            pass
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw)
        tmp_path = Path(fh.name)
    try:
        return load_audio(tmp_path, target_sr=target_sr)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  saving
# --------------------------------------------------------------------------- #
def save_audio(
    path: "str | Path",
    audio: "AudioBuffer | np.ndarray",
    sr: Optional[int] = None,
    subtype: str = "PCM_16",
    peak_limit: float = 0.999,
) -> Path:
    """Write a wav next to a guard against clipping. Returns the path."""
    if isinstance(audio, AudioBuffer):
        samples, sr = audio.samples, audio.sr
    else:
        samples = np.asarray(audio, dtype=np.float32)
        if sr is None:
            raise ValueError("sr is required when passing a raw array")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > peak_limit:
        samples = samples * (peak_limit / peak)

    if module_available("soundfile"):
        import soundfile as sf

        sf.write(str(path), samples, int(sr), subtype=subtype)
        return path

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------- #
#  resampling
# --------------------------------------------------------------------------- #
def resample(audio: AudioBuffer, target_sr: int) -> AudioBuffer:
    if target_sr <= 0 or audio.sr == target_sr or audio.n_samples == 0:
        return audio
    x, sr = audio.samples, audio.sr

    if module_available("soxr"):
        import soxr

        return AudioBuffer(soxr.resample(x, sr, target_sr, quality="HQ"), target_sr)

    if module_available("scipy"):
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target_sr))
        up, down = int(target_sr // g), int(sr // g)
        # keep the polyphase filter reasonable for exotic ratios
        if up < 2000 and down < 2000:
            return AudioBuffer(resample_poly(x, up, down).astype(np.float32), target_sr)

    if module_available("librosa"):
        import librosa

        return AudioBuffer(librosa.resample(x, orig_sr=sr, target_sr=target_sr), target_sr)

    n_out = int(math.floor(len(x) * target_sr / float(sr)))
    idx = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    return AudioBuffer(np.interp(idx, np.arange(len(x)), x).astype(np.float32), target_sr)


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad ``x`` so that ``len(x) == n`` (models change lengths)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


# --------------------------------------------------------------------------- #
#  gain helpers
# --------------------------------------------------------------------------- #
def peak_normalize(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (target / peak)).astype(np.float32) if peak > 1e-9 else x


def rms_normalize(x: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if rms < 1e-9:
        return x
    gain = (10.0 ** (target_db / 20.0)) / rms
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def waveform_preview(audio: AudioBuffer, points: int = 600) -> list:
    """Downsample to `points` peak values in [0,1] for the frontend canvas."""
    x = np.abs(audio.samples)
    if x.size == 0:
        return [0.0] * points
    if x.size < points:
        x = np.pad(x, (0, points - x.size))
    chunk = int(math.ceil(x.size / points))
    padded = np.pad(x, (0, chunk * points - x.size))
    peaks = padded.reshape(points, chunk).max(axis=1)
    top = float(peaks.max()) or 1.0
    return [round(float(v / top), 4) for v in peaks]
