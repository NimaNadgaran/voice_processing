"""Audio regressions: timeline integrity, all backend adapters, and setup ETAs."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.core.types import AudioBuffer
from src.denoising.methods.spectral_gate import builtin_spectral_gate
from src.denoising.methods.wiener_mmse import mmse_lsa


@pytest.mark.parametrize('sr', [8000, 16000, 44100, 48000])
@pytest.mark.parametrize('length', [0, 1, 29, 513])
@pytest.mark.parametrize('method', [builtin_spectral_gate, mmse_lsa])
def test_dsp_short_audio_keeps_shape_and_finite_samples(sr, length, method):
    x = np.random.default_rng(31).normal(0, .02, length).astype(np.float32)
    y, _ = method(x, sr)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_smoothing_does_not_grow_a_short_mask_or_attenuate_edges():
    from src.denoising.methods.spectral_gate import _smooth2d
    x = np.ones((2, 1), np.float32)
    np.testing.assert_allclose(_smooth2d(x, 4, 6), x)


@pytest.mark.parametrize('length', [1, 80, 480, 481, 1003])
@pytest.mark.parametrize('wrapper', [False, True])
def test_rnnoise_flushes_tail_and_aligns_both_interfaces(length, wrapper):
    from src.denoising.methods.rnnoise import RNNoiseDenoiser, ALGORITHMIC_DELAY
    backend = RNNoiseDenoiser()
    destroyed = []
    def create():
        return [np.zeros(ALGORITHMIC_DELAY, np.int16)]
    def process(state, frame):
        combined = np.concatenate((state[0], frame))
        state[0] = combined[len(frame):]
        return combined[:len(frame)], .75
    backend._lowlevel = (create, lambda state: destroyed.append(True), process)
    if wrapper:
        class Wrapper:
            def __init__(self, sr):
                self.state = create()
            def denoise_chunk(self, frame):
                samples, prob = process(self.state, frame)
                yield np.array([[prob]]), samples[None, :]
        backend._factory = Wrapper
    x = np.linspace(.01, .2, length, dtype=np.float32)
    y, _ = (backend._denoise_wrapper if wrapper else backend._denoise_lowlevel)(AudioBuffer(x, 48000))
    np.testing.assert_allclose(y.samples, x, atol=1 / 32767)
    assert len(y.samples) == length
    if not wrapper:
        assert destroyed == [True]


@pytest.mark.parametrize('length', [1, 203, 1003, 2051])
def test_enhancement_chunks_keep_short_tail_and_endpoints(length):
    from src.denoising.chunking import chunked_enhance
    x = np.random.default_rng(5).normal(size=length).astype(np.float32)
    y = chunked_enhance(lambda block: block, x, 100, chunk_s=2, overlap_s=.5, context_s=.2)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_separation_stitches_swapped_speakers_with_a_short_final_block():
    from src.separation.chunking import chunked_separate
    # Independent source waveforms with a known sign in the mixture; an oracle
    # adapter alternates output order at each call to expose stitch errors.
    x = np.arange(1203, dtype=np.float32)
    calls = []
    def process(block):
        sources = [np.sin(block / 7).astype(np.float32), np.cos(block / 11).astype(np.float32)]
        calls.append(len(block))
        return sources if len(calls) % 2 else sources[::-1]
    out = chunked_separate(process, x, 100, chunk_s=3, overlap_s=.5, max_direct_s=0)
    np.testing.assert_allclose(out[0], np.sin(x / 7), atol=1e-5)
    np.testing.assert_allclose(out[1], np.cos(x / 11), atol=1e-5)
    assert all(n == 300 for n in calls)


def test_separation_rejects_changing_source_count():
    from src.separation.chunking import chunked_separate
    count = []
    def process(block):
        count.append(True)
        return [block] if len(count) == 1 else [block, block]
    with pytest.raises(RuntimeError, match='source count'):
        chunked_separate(process, np.ones(300, np.float32), 100, chunk_s=2, overlap_s=.2, max_direct_s=0)


def test_cleanup_keeps_diarization_labels_and_short_quiet_turns():
    from src.separation.base import BaseSeparator
    backend = BaseSeparator()
    backend._loaded = True
    x = np.sin(np.arange(16000) / 10).astype(np.float32) * .1
    quiet = x.copy() * .001
    quiet[3200:] = 0
    backend._separate = lambda audio, n: ([x, np.zeros_like(x), quiet], {
        'labels': ['loud', 'empty', 'quiet'], 'segments': [[[0, 1]], [], [[0, .2]]]})
    result = backend.run(AudioBuffer(x, 16000), 2)
    assert [t.label for t in result.tracks] == ['loud', 'quiet']
    assert result.tracks[1].segments == [[0, .2]]


@pytest.mark.parametrize('shape', [(2, 17), (17, 2), (2, 1, 17), (1, 2, 17), (1, 2, 1, 17)])
def test_clearvoice_keeps_both_sources(shape):
    from src.separation.methods.mossformer_clearvoice import MossFormerClearVoiceSeparator
    sources = MossFormerClearVoiceSeparator._to_sources({'model': {'input.wav': np.ones(shape, np.float32)}})
    assert len(sources) == 2
    assert all(s.shape == (17,) for s in sources)


def wav_blob(sr=16000):
    import soundfile as sf
    out = io.BytesIO()
    sf.write(out, np.sin(np.arange(sr) / 10) * .1, sr, format='WAV')
    return base64.b64encode(out.getvalue()).decode()


def test_cloud_denosing_parses_a_single_json_result():
    from src.denoising.methods.api_huggingface import HuggingFaceAPIDenoiser
    response = SimpleNamespace(headers={'content-type': 'application/json'},
        json=lambda: {'blob': wav_blob(), 'label': 'clean'})
    audio, info = HuggingFaceAPIDenoiser._parse(response, AudioBuffer(np.ones(16000), 16000))
    assert audio.sr == 16000 and info['label'] == 'clean'


def test_cloud_separation_resamples_mixed_rate_responses():
    from src.separation.methods.api_huggingface import HuggingFaceAPISeparator
    response = SimpleNamespace(headers={'content-type': 'application/json'},
        json=lambda: [{'blob': wav_blob(8000)}, {'blob': wav_blob(16000)}])
    sources, labels, sr = HuggingFaceAPISeparator._parse(response)
    assert sr == 8000 and len(sources) == 2
    assert all(len(s) == 8000 for s in sources)


def test_hosted_audio_uses_router_and_independent_wav_payloads(monkeypatch):
    from src.core.hosted_audio import request_audio
    monkeypatch.delenv('HF_DENOISE_URL', raising=False)
    response = SimpleNamespace(status_code=200)
    with patch('requests.post', return_value=response) as post:
        request_audio(AudioBuffer(np.ones(80) * .1, 8000), 'model', 'test-token', 'denoise')
        first = post.call_args.kwargs['data']
        request_audio(AudioBuffer(np.ones(80) * .2, 8000), 'model', 'test-token', 'denoise')
        assert 'router.huggingface.co' in post.call_args.args[0]
        assert first != post.call_args.kwargs['data']
        assert first[:4] == b'RIFF'


def test_hosted_audio_missing_endpoint_explains_deployment():
    from src.core.hosted_audio import request_audio
    from src.core.errors import AppError
    with patch('requests.post', return_value=SimpleNamespace(status_code=404)):
        with pytest.raises(AppError, match='endpoint'):
            request_audio(AudioBuffer(np.zeros(80), 8000), 'model', 'test', 'separation')


def test_overlap_measures_overlapping_turns_even_when_most_of_clip_is_silent():
    from src.separation.diarization import overlap_ratio, merge_turns
    assert overlap_ratio([[[1, 2]], [[1.5, 2.5]]], 10) == .05
    assert merge_turns([[-1, .2], [.1, .8], [1, 10]], 2) == [[0., .8], [1., 2]]


def test_pyannote_v4_and_overlapping_short_turns():
    import torch
    from src.separation.methods.pyannote_diarize import PyannoteDiarizationSeparator
    turns = [(SimpleNamespace(start=1., end=1.2), None, 'a'),
             (SimpleNamespace(start=1.1, end=1.4), None, 'b')]
    annotation = SimpleNamespace(itertracks=lambda **kwargs: iter(turns))
    backend = PyannoteDiarizationSeparator()
    backend._loaded = True
    backend._model = lambda *args, **kwargs: SimpleNamespace(speaker_diarization=annotation)
    result = backend.run(AudioBuffer(np.ones(32000) * .1, 16000), 2)
    assert len(result.tracks) == 2
    assert result.metrics['estimated_overlap_ratio'] == pytest.approx(.05)


def test_nemo_manifest_honors_speaker_hint_and_parses_rttm(tmp_path):
    from omegaconf import OmegaConf
    from src.separation.methods.nemo_msdd import NemoMSDDSeparator
    cfg = OmegaConf.create({'diarizer': {'clustering': {'parameters': {}}}})
    def diarize():
        (tmp_path / 'pred.rttm').write_text('SPEAKER input 1 0.5 1.0 <NA> <NA> speaker_0 <NA> <NA>\n')
    backend = NemoMSDDSeparator()
    backend._model = SimpleNamespace(cfg=cfg, diarize=diarize)
    result = backend._run(tmp_path / 'input.wav', tmp_path, 3)
    assert result == {'speaker_0': [[.5, 1.5]]}
    assert cfg.diarizer.clustering.parameters.oracle_num_speakers
    assert cfg.diarizer.clustering.parameters.max_num_speakers == 3
    assert json.loads((tmp_path / 'manifest.json').read_text())['num_speakers'] == 3


@pytest.mark.parametrize('method', ['sepformer', 'convtasnet_asteroid', 'local_convtasnet'])
def test_neural_separation_adapters_handle_tiny_clips(method):
    import torch
    from src.core.registry import get_separator
    backend = type(get_separator(method))()
    backend._loaded = True
    backend._device = 'cpu'
    def outputs(tensor):
        return torch.stack((tensor, tensor * .5), dim=1)
    if method == 'sepformer':
        model = SimpleNamespace(hparams=SimpleNamespace(sample_rate=8000, num_spks=2),
            separate_batch=lambda tensor: outputs(tensor).transpose(1, 2))
        backend._get_model = lambda model_id: (model, 'cpu')
    elif method == 'convtasnet_asteroid':
        model = SimpleNamespace(sample_rate=8000, n_src=2, separate=outputs)
        backend._get_model = lambda model_id: (model, 'cpu')
    else:
        from src.separation.architectures import build_model
        backend._model = build_model(dict(enc_channels=8, bottleneck=4, hidden=8, n_blocks=1, n_repeats=1))
        backend._sr = 8000
    result = backend.run(AudioBuffer(np.ones(3, np.float32) * .1, 48000), 2)
    assert result.tracks
    assert all(t.audio.sr == 48000 and len(t.audio.samples) == 3 for t in result.tracks)


def test_dns64_and_unet_chunk_adapters():
    import torch
    from src.denoising.methods.demucs_denoiser import DemucsDenoiser
    from src.denoising.methods.local_unet import LocalUNetDenoiser
    x = np.sin(np.arange(3103) / 10).astype(np.float32) * .1
    for backend in (DemucsDenoiser(), LocalUNetDenoiser()):
        backend._device = 'cpu'
        class Identity:
            n_fft = 32
            def __call__(self, tensor):
                return tensor.clone()
        backend._model = Identity()
        out, _ = backend._denoise(AudioBuffer(x, 100))
        np.testing.assert_allclose(out.samples, x, atol=1e-6)


def test_demucs_vocals_preserves_amplitude_and_handles_silence():
    import torch
    from src.denoising.methods.demucs_vocals import DemucsVocalsDenoiser
    backend = DemucsVocalsDenoiser()
    backend._device = 'cpu'
    backend._model = SimpleNamespace(audio_channels=2, sources=['drums', 'bass', 'other', 'vocals'])
    x = np.sin(np.arange(4000) / 10).astype(np.float32) * .1
    def apply(model, tensor, **kwargs):
        return torch.stack([tensor * 0, tensor * 0, tensor * 0, tensor], dim=1)
    with patch('demucs.apply.apply_model', side_effect=apply):
        out, _ = backend._denoise(AudioBuffer(x, 44100))
        np.testing.assert_allclose(out.samples, x, atol=1e-6)
        out, info = backend._denoise(AudioBuffer(np.zeros_like(x), 44100))
        assert not out.samples.any() and info['silent_input']


def test_resemble_processes_once_and_reports_non_generative_fallback():
    import torch
    import sys
    from src.denoising.methods.resemble_enhance import ResembleEnhanceDenoiser
    backend = ResembleEnhanceDenoiser()
    backend._device = 'cpu'
    audio = AudioBuffer(np.ones(100) * .1, 44100)
    inference = SimpleNamespace(enhance=Mock(side_effect=lambda x, sr, device, **kw: (x, sr)),
                                denoise=Mock(side_effect=lambda x, sr, device: (x, sr)))
    with patch.dict(sys.modules, {'resemble_enhance.enhancer.inference': inference}):
        _, info = backend._denoise(audio)
        assert info['generative'] and inference.enhance.call_count == 1
        inference.denoise.assert_not_called()
        inference.enhance.side_effect = RuntimeError('enhancement failed')
        _, info = backend._denoise(audio)
        assert not info['generative'] and info['enhancement_error'] == 'enhancement failed'


def test_timing_probe_restores_weights_buffers_optimizer_and_rng():
    import torch
    from src.core.training_runtime import preserve_training_state
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with preserve_training_state(model, optimizer, scaler):
        loss = model(torch.randn(4, 2)).square().mean()
        loss.backward()
        optimizer.step()
    assert not optimizer.state
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_training_estimate_counts_validation_and_remaining_epochs(tmp_path):
    import torch
    from src.core.training_runtime import measured_estimate
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    cfg = dict(steps_per_epoch=100, epochs=10)
    path = tmp_path / 'eta.json'
    with patch('src.core.training_runtime.time.perf_counter', side_effect=[10., 12.]):
        result = measured_estimate(model, optimizer, scaler, lambda: .5,
            lambda batches: list(batches), [None] * 9, cfg, start_epoch=4, path=path)
    assert result['validation_seconds_per_epoch'] == 6
    assert result['remaining_seconds'] == 6 * (50 + 6)
    assert json.loads(path.read_text())['remaining_seconds'] == 336


def test_setup_defaults_to_all_local_models_and_supports_selection():
    from setup import select_models
    assert select_models() == ['local_unet', 'local_convtasnet']
    assert select_models('denoise,local_unet') == ['local_unet']
    with pytest.raises(ValueError, match='Unknown local model'):
        select_models('deepfilternet')


def test_rnnoise_modern_frame_interface_avoids_resampling_graph_and_resets():
    from src.denoising.methods.rnnoise import RNNoiseDenoiser, ALGORITHMIC_DELAY
    instances = []
    class Wrapper:
        def __init__(self, sr):
            self.channels = None
            self.dtype = None
            self.pending = np.zeros(ALGORITHMIC_DELAY, np.int16)
            self.reset_called = False
            instances.append(self)
        def denoise_chunk(self, frame):
            raise AssertionError('unnecessary resampling graph')
        def denoise_frame(self, frame):
            assert self.channels == 1 and frame.shape == (1, 480)
            combined = np.concatenate((self.pending, frame[0]))
            self.pending = combined[480:]
            return np.array([[.75]]), combined[:480][None, :]
        def reset(self):
            self.reset_called = True
    backend = RNNoiseDenoiser()
    backend._factory = Wrapper
    x = np.linspace(.01, .2, 481, dtype=np.float32)
    out, _ = backend._denoise_wrapper(AudioBuffer(x, 48000))
    np.testing.assert_allclose(out.samples, x, atol=1 / 32767)
    assert instances[0].reset_called


def test_asteroid_published_rate_wins_over_missing_checkpoint_metadata(monkeypatch):
    import torch
    from src.separation.methods.convtasnet_asteroid import AsteroidConvTasNetSeparator
    monkeypatch.delenv('ASTEROID_MODEL', raising=False)
    backend = AsteroidConvTasNetSeparator()
    sizes = []
    def separate(tensor):
        sizes.append(tensor.shape[-1])
        return torch.stack((tensor, tensor * .5), dim=1)
    backend._get_model = lambda model_id: (SimpleNamespace(sample_rate=8000, n_src=2, separate=separate), 'cpu')
    sources, info = backend._separate(AudioBuffer(np.ones(16000, np.float32) * .1, 16000), 2)
    assert sizes == [16000]
    assert info['output_sr'] == 16000
    assert all(len(source) == 16000 for source in sources)


@pytest.mark.parametrize('kind', ['clean', 'speakers'])
def test_setup_rejects_explicit_missing_data_without_downloading(tmp_path, kind):
    from setup import prepare_data
    with patch('setup.download_corpus') as download:
        with pytest.raises(FileNotFoundError, match='supplied'):
            prepare_data(['local_unet', 'local_convtasnet'], **{f'{kind}_dir': tmp_path / 'missing'})
        download.assert_not_called()


@pytest.mark.parametrize('kind', ['denoise', 'separation'])
def test_cloud_token_alone_does_not_promise_hosted_inference(monkeypatch, kind):
    from src.denoising.methods.api_huggingface import HuggingFaceAPIDenoiser
    from src.separation.methods.api_huggingface import HuggingFaceAPISeparator
    monkeypatch.setenv('HF_TOKEN', 'test-token')
    prefix = 'HF_DENOISE' if kind == 'denoise' else 'HF_SEPARATION'
    monkeypatch.delenv(prefix + '_MODEL', raising=False)
    monkeypatch.delenv(prefix + '_URL', raising=False)
    backend = HuggingFaceAPIDenoiser() if kind == 'denoise' else HuggingFaceAPISeparator()
    available, reason = backend.check_available()
    assert not available and 'deploy' in reason


@pytest.mark.parametrize('backend_length', [2, 20])
def test_direct_separator_enforces_original_length(backend_length):
    from src.separation.chunking import chunked_separate
    out = chunked_separate(lambda block: [np.ones(backend_length)], np.ones(10), 100)
    assert len(out[0]) == 10


def test_overlap_preset_does_not_suppress_a_second_voice_with_single_speaker_denoising():
    from src.pipeline.paths import PRESET_PATHS
    path = next(path for path in PRESET_PATHS if path.id == 'path3')
    assert path.denoiser == 'wiener_mmse'
    assert path.separator == 'convtasnet_asteroid'
    assert path.separator_alts == ['sepformer']
