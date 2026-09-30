"""Smoke tests -- no optional dependency, no network, no training.

    python -m pytest tests/ -v
    python tests/test_smoke.py          # also works without pytest

Everything here must pass on a bare ``pip install -r requirements.txt``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.audio_io import load_audio, resample, save_audio, waveform_preview  # noqa: E402
from src.core.dsp import energy_vad, istft, mask_to_segments, mfcc, stft  # noqa: E402
from src.core.metrics import denoise_metrics, estimated_snr, separation_metrics, si_sdr  # noqa: E402
from src.core.registry import list_denoisers, list_separators  # noqa: E402
from src.core.types import AudioBuffer  # noqa: E402
from src.denoising import denoise  # noqa: E402
from src.separation import count_speakers, separate  # noqa: E402

SR = 16000


# --------------------------------------------------------------------------- #
def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _two_speaker_clip(seconds: float = 8.0, sr: int = SR):
    """Turn-taking 'conversation': two very different pitched voices + noise."""
    rng = np.random.default_rng(0)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    def voice(f0, formant):
        sig = np.zeros(n, dtype=np.float32)
        for k in range(1, 12):
            sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)
        env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
        colour = np.sin(2 * np.pi * formant * t).astype(np.float32) * 0.3
        return ((sig + colour) * env).astype(np.float32)

    a, b = voice(110.0, 700.0), voice(230.0, 1600.0)
    gate = np.zeros(n, dtype=np.float32)
    gate[: n // 2] = 1.0
    mix = a * gate + b * (1 - gate)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.6
    noise = rng.standard_normal(n).astype(np.float32) * 0.05
    return AudioBuffer(mix + noise, sr), AudioBuffer(mix, sr)


# --------------------------------------------------------------------------- #
#  core
# --------------------------------------------------------------------------- #
def test_audio_buffer_basics():
    buf = AudioBuffer(_tone(440, 1.0), SR)
    assert buf.n_samples == SR
    assert abs(buf.duration - 1.0) < 1e-6
    assert 0.2 < buf.peak() < 0.4
    stereo = AudioBuffer(np.stack([_tone(440, 0.5), _tone(880, 0.5)]), SR)
    assert stereo.samples.ndim == 1  # collapsed to mono


def test_io_roundtrip():
    buf = AudioBuffer(_tone(440, 0.5), SR)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.wav"
        save_audio(path, buf)
        back = load_audio(path)
        assert back.sr == SR
        assert abs(back.n_samples - buf.n_samples) <= 1
        assert np.corrcoef(back.samples[:1000], buf.samples[:1000])[0, 1] > 0.99


def test_resample_preserves_duration():
    buf = AudioBuffer(_tone(440, 1.0), SR)
    for target in (8000, 22050, 48000):
        out = resample(buf, target)
        assert out.sr == target
        assert abs(out.duration - 1.0) < 0.02


def test_stft_istft_roundtrip():
    x = _tone(440, 1.0) + 0.1 * _tone(1000, 1.0)
    spec = stft(x, n_fft=1024, hop=256)
    back = istft(spec, hop=256, length=len(x))
    assert len(back) == len(x)
    # ignore the edges where the window taper dominates
    assert np.corrcoef(back[2000:-2000], x[2000:-2000])[0, 1] > 0.99


def test_vad_finds_speech_and_silence():
    sr = SR
    signal = np.concatenate([np.zeros(sr, np.float32), _tone(300, 1.0), np.zeros(sr, np.float32)])
    mask, hop = energy_vad(signal, sr)
    segments = mask_to_segments(mask, hop)
    assert segments, "VAD found nothing"
    start, end = segments[0]
    assert 0.6 < start < 1.4 and 1.7 < end < 2.4


def test_mfcc_shape():
    feats = mfcc(_tone(440, 1.0), SR, n_mfcc=20, n_fft=512, hop=160)
    assert feats.shape[0] == 20 and feats.shape[1] > 50
    assert np.isfinite(feats).all()


def test_waveform_preview_shape():
    peaks = waveform_preview(AudioBuffer(_tone(440, 3.0), SR), points=200)
    assert len(peaks) == 200
    assert all(0.0 <= p <= 1.0 for p in peaks)


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def test_si_sdr_is_scale_invariant():
    ref = _tone(440, 1.0)
    assert si_sdr(ref * 3.0, ref) > 60          # perfect up to scale
    assert si_sdr(np.random.randn(len(ref)).astype(np.float32), ref) < 5


def test_estimated_snr_orders_correctly():
    clean = AudioBuffer(_tone(300, 2.0), SR)
    rng = np.random.default_rng(0)
    noisy = AudioBuffer(clean.samples + rng.standard_normal(clean.n_samples).astype(np.float32) * 0.2, SR)
    assert estimated_snr(clean)["snr_db"] > estimated_snr(noisy)["snr_db"]


def test_denoise_metrics_keys():
    noisy, clean = _two_speaker_clip(4.0)
    metrics = denoise_metrics(noisy, clean)
    for key in ("snr_improvement_db", "noise_reduction_db", "speech_preserved", "quality_score"):
        assert key in metrics
    assert 0.0 <= metrics["speech_preserved"] <= 1.0
    assert 0.0 <= metrics["quality_score"] <= 100.0


def test_separation_metrics_detect_duplicates():
    _, clean = _two_speaker_clip(4.0)
    duplicated = separation_metrics(clean, [clean, clean])
    assert duplicated["mean_cross_correlation"] > 0.9      # identical tracks
    half = clean.n_samples // 2
    a = AudioBuffer(np.concatenate([clean.samples[:half], np.zeros(half, np.float32)]), clean.sr)
    b = AudioBuffer(np.concatenate([np.zeros(half, np.float32), clean.samples[half:]]), clean.sr)
    disjoint = separation_metrics(clean, [a, b])
    assert disjoint["mean_cross_correlation"] < duplicated["mean_cross_correlation"]
    assert disjoint["energy_conservation"] > 0.9


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #
def test_registry_lists_everything():
    denoisers = {m.key for m in list_denoisers()}
    separators = {m.key for m in list_separators()}
    assert {"none", "spectral_gate", "wiener_mmse", "deepfilternet", "local_unet"} <= denoisers
    assert {"diarize_cluster", "sepformer", "pyannote", "local_convtasnet"} <= separators


def test_always_available_methods_are_available():
    available = {m.key for m in list_denoisers() if m.available}
    assert {"none", "spectral_gate", "wiener_mmse"} <= available
    assert "diarize_cluster" in {m.key for m in list_separators() if m.available}


def test_unavailable_methods_explain_themselves():
    for info in list_denoisers() + list_separators():
        if not info.available:
            assert info.unavailable_reason, "%s must say why it is unavailable" % info.key


# --------------------------------------------------------------------------- #
#  denoising
# --------------------------------------------------------------------------- #
def test_spectral_gate_reduces_noise():
    noisy, _ = _two_speaker_clip(6.0)
    result = denoise(noisy, method="spectral_gate")
    assert not result.metrics.get("error"), result.metrics.get("error")
    assert result.audio.n_samples == noisy.n_samples
    assert result.audio.sr == noisy.sr
    assert result.metrics["noise_reduction_db"] > 3.0
    assert result.metrics["speech_preserved"] > 0.5
    assert np.isfinite(result.audio.samples).all()


def test_wiener_mmse_runs():
    noisy, _ = _two_speaker_clip(4.0)
    result = denoise(noisy, method="wiener_mmse")
    assert not result.metrics.get("error")
    assert result.audio.n_samples == noisy.n_samples
    assert np.isfinite(result.audio.samples).all()


def test_passthrough_is_identical():
    noisy, _ = _two_speaker_clip(3.0)
    result = denoise(noisy, method="none")
    assert np.allclose(result.audio.samples, noisy.samples)


def _an_unavailable_denoiser():
    """Any denoiser this machine genuinely cannot run.

    Deliberately *not* hardcoded: which methods are missing depends on what is
    pip-installed and, for local_unet, on whether a checkpoint has been trained
    yet.  Training the local model used to break these tests.
    """
    from src.core.registry import list_denoisers

    for info in list_denoisers():
        if not info.available:
            return info.key
    return None


def test_unavailable_denoiser_fails_softly():
    method = _an_unavailable_denoiser()
    if method is None:
        return  # every backend is installed here -- nothing to fail softly
    noisy, _ = _two_speaker_clip(2.0)
    result = denoise(noisy, method=method, safe=True)
    assert result.metrics.get("error")                        # reported, not raised
    assert result.audio.n_samples == noisy.n_samples          # input passed through


# --------------------------------------------------------------------------- #
#  separation
# --------------------------------------------------------------------------- #
def test_speaker_count_returns_sane_shape():
    noisy, _ = _two_speaker_clip(10.0)
    est = count_speakers(noisy)
    assert 1 <= est["n_speakers"] <= 8
    assert 0.0 <= est["confidence"] <= 1.0
    assert est["method"] in {"mfcc-numpy", "ecapa-tdnn", "too-short"}


def test_separation_produces_requested_tracks():
    noisy, _ = _two_speaker_clip(10.0)
    result = separate(noisy, method="diarize_cluster", num_speakers=2)
    assert not result.metrics.get("error"), result.metrics.get("error")
    assert len(result.tracks) >= 1
    for track in result.tracks:
        assert track.audio.sr == noisy.sr
        assert track.audio.n_samples == noisy.n_samples     # timeline aligned
        assert np.isfinite(track.audio.samples).all()
        assert track.total_speech >= 0.0


def test_separation_writes_files():
    noisy, _ = _two_speaker_clip(8.0)
    with tempfile.TemporaryDirectory() as tmp:
        result = separate(noisy, method="diarize_cluster", num_speakers=2, output_dir=tmp)
        files = sorted(Path(tmp).glob("*.wav"))
        assert len(files) == len(result.tracks) >= 1
        for f in files:
            assert load_audio(f).n_samples > 0


# --------------------------------------------------------------------------- #
#  pipeline
# --------------------------------------------------------------------------- #
def test_full_pipeline_two_paths():
    from src.pipeline import PipelineOptions, resolve_path, run_pipeline

    noisy, _ = _two_speaker_clip(10.0)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.wav"
        save_audio(source, noisy)

        events = []
        report = run_pipeline(
            source,
            [resolve_path({"id": "path1"}), resolve_path({"id": "path8"})],
            PipelineOptions(num_speakers=2),
            progress=events.append,
        )

    assert report["input"]["duration"] > 9
    assert len(report["paths"]) == 2
    for path in report["paths"]:
        assert path["status"] == "ok", path.get("error")
        assert Path(path["denoised"]["path"]).exists()
        assert path["tracks"], "no speaker tracks"
        for track in path["tracks"]:
            assert Path(track["path"]).exists()
    assert report["comparison"]["ranking"]
    assert any(e.get("artifact") for e in events), "no artifact events for the live UI"


def test_pipeline_reports_failure_without_raising():
    from src.pipeline import PipelineOptions, resolve_path, run_pipeline

    method = _an_unavailable_denoiser()
    if method is None:
        return  # every backend is installed here -- no failure to report
    noisy, _ = _two_speaker_clip(3.0)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.wav"
        save_audio(source, noisy)
        report = run_pipeline(
            source,
            [resolve_path({"denoiser": method, "separator": "diarize_cluster"})],
            PipelineOptions(),
        )
    assert report["paths"][0]["status"] == "failed"
    assert "error" in report["paths"][0]


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except Exception as exc:
            failures += 1
            print("  FAIL  %s -> %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
