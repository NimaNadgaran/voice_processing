"""Stage-3 regressions; model adapters are faked to avoid network in unit tests."""
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.types import AudioBuffer
from src.pipeline.runner import PipelineOptions, PipelineRunner
from src.pipeline.paths import PipelinePath, resolve_path
from src.transcription import transcribe, resolve_method
from src.transcription.base import BaseTranscriber, TranscriptResult, speech_windows, write_transcript
from src.transcription.languages import normalize_language


@pytest.mark.parametrize('value,expected', [('Persian', 'fa'), ('Farsi', 'fa'), ('فارسی', 'fa'),
    ('fa-IR', 'fa'), ('FA_IR', 'fa'), ('English', 'en'), ('ar', 'ar'), ('auto', None), (None, None)])
def test_language_codes_and_aliases(value, expected):
    assert normalize_language(value) == expected


@pytest.mark.parametrize('value', ['made-up', '../fa', '<script>', 'xx'])
def test_unknown_language_is_rejected(value):
    with pytest.raises(ValueError, match='Unknown speech language'):
        normalize_language(value)


def fake_backend():
    from src.core.types import MethodInfo
    backend = BaseTranscriber()
    backend.info = MethodInfo('test_asr', 'Test', 'transcribe', 'deep-pretrained', '',
        supported_languages=['fa', 'en'], supports_auto_language=True)
    backend.load = Mock()
    backend._recognize = Mock(return_value={'text': 'سلام دنیا', 'language': 'fa',
        'segments': [{'start': 0., 'end': .5, 'text': 'سلام دنیا'}]})
    return backend


def test_silent_or_empty_audio_does_not_load_a_model():
    backend = fake_backend()
    for length, regions in [(0, None), (16000, None), (16000, [])]:
        result = backend.run(AudioBuffer(np.zeros(length), 16000), regions=regions)
        assert result.status == 'empty' and result.text == ''
    backend.load.assert_not_called()
    backend._recognize.assert_not_called()


def test_turn_recognition_preserves_original_timestamps_and_language():
    backend = fake_backend()
    result = backend.run(AudioBuffer(np.ones(160000) * .1, 16000), 'fa', [[2., 3.], [7., 8.]])
    assert result.language == 'fa' and result.text == 'سلام دنیا\nسلام دنیا'
    assert result.segments[0]['start'] == pytest.approx(1.82)
    assert result.segments[1]['start'] == pytest.approx(6.82)
    assert all(call.args[1] == 'fa' for call in backend._recognize.call_args_list)


def test_long_turns_are_bounded_without_duplicate_or_missing_samples():
    audio = AudioBuffer(np.ones(16000 * 73) * .1, 16000)
    chunks = list(speech_windows(audio))
    assert len(chunks) >= 3
    assert max(len(samples) for _, samples in chunks) <= 28 * 16000
    assert sum(len(samples) for _, samples in chunks) == audio.n_samples
    for (offset, samples), (next_offset, _) in zip(chunks, chunks[1:]):
        assert next_offset == pytest.approx(offset + len(samples) / 16000)


def test_padding_merges_adjacent_turns_instead_of_transcribing_twice():
    chunks = list(speech_windows(AudioBuffer(np.ones(80000) * .1, 16000), [[1, 2], [2.1, 3]]))
    assert len(chunks) == 1


def test_repetitive_recognition_is_flagged_without_discarding_possible_real_repetition():
    backend = fake_backend()
    backend._recognize.return_value = {'text': 'hello ' * 50, 'language': 'en'}
    result = backend.run(AudioBuffer(np.ones(16000) * .1, 16000), 'en')
    assert result.status == 'ok' and result.text.startswith('hello')
    assert 'Highly repetitive' in result.metrics['warnings'][0]


@pytest.mark.parametrize('samples,sr', [(np.array([np.nan]), 16000), (np.array([.1]), 0)])
def test_invalid_audio_is_not_sent_to_model(samples, sr):
    backend = fake_backend()
    with pytest.raises(ValueError, match='finite audio'):
        backend.run(AudioBuffer(samples, sr))
    backend.load.assert_not_called()


def test_persian_specialist_rejects_english_and_auto_assumes_persian():
    from src.transcription.methods.persian_wav2vec2 import PersianWav2Vec2Transcriber
    backend = PersianWav2Vec2Transcriber()
    assert backend.validate_language('auto') == 'fa'
    with pytest.raises(ValueError, match='does not support en'):
        backend.validate_language('en')


def test_vosk_needs_explicit_language():
    from src.transcription.methods.vosk import VoskTranscriber
    with pytest.raises(ValueError, match='explicit language'):
        VoskTranscriber().validate_language('auto')


def test_faster_whisper_exhausts_generator_and_uses_transcribe_not_translate():
    from src.transcription.methods.faster_whisper import FasterWhisperTranscriber
    backend = FasterWhisperTranscriber()
    generator = (SimpleNamespace(start=1., end=2., text=' سلام ') for _ in range(2))
    backend._model = SimpleNamespace(supported_languages=['fa', 'en'],
        transcribe=Mock(return_value=(generator, SimpleNamespace(language='fa'))))
    result = backend._recognize(np.ones(16000, np.float32), 'fa')
    assert result['text'] == 'سلام سلام' and len(result['segments']) == 2
    assert backend._model.transcribe.call_args.kwargs['task'] == 'transcribe'
    assert backend._model.transcribe.call_args.kwargs['language'] == 'fa'


def test_faster_whisper_english_only_checkpoint_cannot_accept_persian():
    from src.transcription.methods.faster_whisper import FasterWhisperTranscriber
    backend = FasterWhisperTranscriber()
    backend._model = SimpleNamespace(supported_languages=['en'])
    with pytest.raises(ValueError, match='does not support fa'):
        backend._recognize(np.ones(16000, np.float32), 'fa')


def test_faster_whisper_cuda_generator_failure_has_cpu_fix():
    from src.transcription.methods.faster_whisper import FasterWhisperTranscriber
    from src.core.errors import AppError
    backend = FasterWhisperTranscriber()
    def failed_segments():
        raise RuntimeError('Library cublas64_12.dll is not found')
        yield  # model failures can occur only when the lazy generator is read
    backend._model = SimpleNamespace(supported_languages=['en'],
        transcribe=lambda *args, **kwargs: (failed_segments(), SimpleNamespace(language='en')))
    with pytest.raises(AppError) as caught:
        backend._recognize(np.ones(16000, np.float32), 'en')
    assert 'STT_DEVICE=cpu' in caught.value.fix


def test_hf_whisper_language_switches_do_not_reuse_a_previous_language():
    import torch
    from src.transcription.methods.whisper_transformers import TransformersWhisperTranscriber
    backend = TransformersWhisperTranscriber()
    backend._device = 'cpu'
    backend._processor = Mock()
    backend._processor.return_value = {'input_features': torch.ones(1, 80, 3000), 'attention_mask': torch.ones(1, 3000)}
    backend._processor.batch_decode.side_effect = lambda tokens, skip_special_tokens, **kwargs: ['سلام' if skip_special_tokens else '<|fa|>سلام']
    pinned_config = SimpleNamespace(language='fa', forced_decoder_ids=[[1, 4]])
    backend._model = SimpleNamespace(generation_config=pinned_config,
        generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([[1, 2]]))))
    for language in ('fa', 'en', None):
        result = backend._recognize(np.ones(8000), language)
        assert backend._model.generate.call_args.kwargs['language'] == language
        assert backend._model.generate.call_args.kwargs['task'] == 'transcribe'
        assert backend._model.generate.call_args.kwargs['generation_config'].language == language
        assert backend._model.generate.call_args.kwargs['generation_config'].forced_decoder_ids is None
        assert backend._model.generate.call_args.kwargs['no_repeat_ngram_size'] == 6
    assert result['language'] == 'fa'
    assert pinned_config.language == 'fa' and pinned_config.forced_decoder_ids == [[1, 4]]


def test_persian_ctc_adapter_resamples_contract_and_decodes_argmax():
    import torch
    from src.transcription.methods.persian_wav2vec2 import PersianWav2Vec2Transcriber
    backend = PersianWav2Vec2Transcriber()
    backend._device = 'cpu'
    backend._processor = Mock()
    backend._processor.return_value = {'input_values': torch.ones(1, 800)}
    backend._processor.batch_decode.return_value = ['سلام']
    backend._model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[0., 1.], [1., 0.]]])))
    assert backend._recognize(np.ones(3), 'fa')['text'] == 'سلام'
    assert backend._processor.call_args.kwargs['sampling_rate'] == 16000
    assert len(backend._processor.call_args.args[0]) == 800


def test_hf_whisper_english_only_model_is_supported_without_language_task_prompt():
    import torch
    from src.transcription.methods.whisper_transformers import TransformersWhisperTranscriber
    backend = TransformersWhisperTranscriber()
    backend._device = 'cpu'
    backend._processor = Mock(return_value={'input_features': torch.ones(1, 80, 3000)})
    backend._processor.batch_decode.return_value = ['hello']
    backend._model = SimpleNamespace(generation_config=SimpleNamespace(is_multilingual=False),
        generate=Mock(return_value=SimpleNamespace(sequences=torch.tensor([[1, 2]]))))
    for language in (None, 'en'):
        result = backend._recognize(np.ones(8000), language)
        assert result['language'] == 'en'
        assert backend._model.generate.call_args.kwargs['task'] is None
        assert backend._model.generate.call_args.kwargs['language'] is None
        assert backend._model.generate.call_args.kwargs['generation_config'].language is None
    with pytest.raises(ValueError, match='only supports English'):
        backend._recognize(np.ones(8000), 'fa')


def test_auto_never_uses_persian_for_unknown_language():
    def get(key):
        return SimpleNamespace(describe=lambda: SimpleNamespace(available=key == 'persian_wav2vec2'))
    with patch('src.transcription.get_transcriber', side_effect=get):
        assert resolve_method('auto', 'fa') == 'persian_wav2vec2'
        with pytest.raises(RuntimeError, match='No installed'):
            resolve_method('auto', 'auto')
        with pytest.raises(RuntimeError, match='No installed'):
            resolve_method('auto', 'en')


def test_explicit_model_failure_is_reported_without_silent_algorithm_switch():
    backend = SimpleNamespace(safe_run=lambda *args: TranscriptResult('faster_whisper', 'failed', error='cannot load'))
    with patch('src.transcription.get_transcriber', return_value=backend):
        result = transcribe(AudioBuffer(np.ones(16000), 16000), 'faster_whisper')
    assert result.status == 'failed' and result.method == 'faster_whisper'


def test_public_safe_transcribe_covers_file_loading_and_export_errors():
    with patch('src.transcription.load_audio', side_effect=FileNotFoundError('gone.wav')):
        result = transcribe('gone.wav')
        assert result.status == 'failed' and 'found' in result.error
        with pytest.raises(FileNotFoundError):
            transcribe('gone.wav', safe=False)
    backend = SimpleNamespace(safe_run=lambda *args: TranscriptResult('test', text='hello'))
    with patch('src.transcription.get_transcriber', return_value=backend), \
         patch('src.transcription.write_transcript', side_effect=PermissionError('not writable')):
        result = transcribe(AudioBuffer(np.ones(16000), 16000), 'faster_whisper', output_path='test.txt')
        assert result.status == 'failed' and 'permission' in result.error


def test_disabled_transcription_skips_file_load_and_export():
    with patch('src.transcription.load_audio') as load, patch('src.transcription.write_transcript') as write:
        result = transcribe('anything.wav', method='none', output_path='unused.txt')
    assert result.status == 'disabled'
    load.assert_not_called()
    write.assert_not_called()


@pytest.mark.parametrize('options', [{'transcription_method': 'missing'}, {'transcription_language': 'xx'}])
def test_pipeline_rejects_invalid_transcription_configuration(options):
    with pytest.raises(ValueError):
        PipelineOptions(**options)


def test_custom_and_preset_paths_accept_transcription_override():
    assert resolve_path({'id': 'path1', 'transcriber': 'vosk'}).transcriber == 'vosk'
    assert resolve_path({'id': 'custom_stt', 'denoiser': 'none', 'separator': 'none', 'transcriber': 'faster_whisper'}).transcriber == 'faster_whisper'
    with pytest.raises(KeyError):
        resolve_path({'id': 'path1', 'transcriber': 'not_a_model'})


def test_pipeline_exports_persian_transcript_but_keeps_audio_after_one_failure(tmp_path):
    from src.core.audio_io import save_audio
    audio = AudioBuffer(np.sin(np.arange(32000) / 20).astype(np.float32) * .1, 16000)
    source = tmp_path / 'meeting.wav'
    save_audio(source, audio)
    backend = fake_backend()
    result = backend.run(audio, 'fa')
    paths = [PipelinePath('good', 'Good', 'none', 'none'),
             PipelinePath('bad_text', 'Bad text', 'none', 'none', transcriber='vosk')]
    def recognize(audio, method, **kwargs):
        return result if method == 'faster_whisper' else TranscriptResult('vosk', 'failed', error='recognizer failed')
    with patch('src.transcription.transcribe', side_effect=recognize):
        report = PipelineRunner(job_id='test_stt', out_root=tmp_path).run(source, paths,
            PipelineOptions(num_speakers=1, transcription_method='faster_whisper', transcription_language='fa'))
    good, bad = report['paths']
    assert good['status'] == bad['status'] == 'ok'
    data = good['tracks'][0]['transcription']
    assert data['language'] == 'fa'
    assert (tmp_path / 'test_stt/good' / data['file']).read_text(encoding='utf-8').strip() == 'سلام دنیا'
    assert bad['transcription']['status'] == 'failed'
    assert (tmp_path / 'test_stt/bad_text' / bad['tracks'][0]['file']).exists()
    assert 'file' not in bad['tracks'][0]['transcription']


def test_api_serves_utf8_text_and_zip_includes_each_speakers_transcript(tmp_path):
    from src.api import server
    folder = tmp_path / 'job_text/path1'
    folder.mkdir(parents=True)
    write_transcript(folder / 'speaker01.txt', TranscriptResult('test', text='سلام دنیا\nمتن فارسی'))
    with patch.object(server, 'OUT_DIR', tmp_path):
        client = TestClient(server.app)
        response = client.get('/api/files/job_text/path1/speaker01.txt')
        assert response.status_code == 200 and response.text == 'سلام دنیا\nمتن فارسی\n'
        assert response.headers['content-type'] == 'text/plain; charset=utf-8'
        assert 'attachment' in response.headers['content-disposition']
        archive = client.get('/api/jobs/job_text/zip?path_id=path1')
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
            assert zf.read('speaker01.txt').decode('utf-8').startswith('سلام دنیا')


def test_transcript_export_failure_keeps_speaker_audio_and_report(tmp_path):
    from src.core.audio_io import save_audio
    source = tmp_path / 'meeting.wav'
    save_audio(source, AudioBuffer(np.ones(16000) * .1, 16000))
    with patch('src.transcription.transcribe', return_value=TranscriptResult('test_asr', text='hello')), \
         patch('src.transcription.base.write_transcript', side_effect=OSError('disk write failed')):
        report = PipelineRunner(job_id='export_failure', out_root=tmp_path).run(source,
            [PipelinePath('control', 'Control', 'none', 'none')],
            PipelineOptions(num_speakers=1, transcription_method='faster_whisper'))
    record = report['paths'][0]
    assert record['status'] == 'ok' and record['transcription']['status'] == 'failed'
    assert record['tracks'][0]['transcription']['status'] == 'failed'
    assert (tmp_path / 'export_failure/control' / record['tracks'][0]['file']).is_file()
    assert (tmp_path / 'export_failure/report.json').is_file()


def test_api_lists_all_recognizers_language_support_and_persian():
    from src.api import server
    data = server.methods()
    assert {m['key'] for m in data['transcribers']} == {'faster_whisper', 'whisper_transformers', 'persian_wav2vec2', 'vosk'}
    assert next(m for m in data['transcribers'] if m['key'] == 'persian_wav2vec2')['supported_languages'] == ['fa']
    assert any(language['code'] == 'fa' and language['rtl'] for language in data['languages'])
