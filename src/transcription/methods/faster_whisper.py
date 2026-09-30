"""Efficient multilingual Whisper using CTranslate2 (CPU INT8 by default)."""
import os

from ...core.registry import register_transcriber
from ...core.types import MethodInfo
from ...core.errors import AppError
from ...core.utils import PRETRAINED_DIR
from ..base import BaseTranscriber
from ..languages import WHISPER_LANGUAGES


@register_transcriber
class FasterWhisperTranscriber(BaseTranscriber):
    info = MethodInfo(
        'faster_whisper', 'Faster Whisper — multilingual', 'transcribe', 'deep-pretrained',
        'Multilingual speech recognition, including Persian. CPU-friendly INT8 inference; '
        'choose larger Whisper models for quality versus speed.', speed='fast', quality=4,
        pip=['faster_whisper', 'ctranslate2'], install_hint='pip install faster-whisper',
        notes='Local recognition after weights download. Default: small (~500 MB). '
              'STT_WHISPER_MODEL can select tiny/base/medium/large-v3/turbo or a local CT2 directory.',
        supported_languages=WHISPER_LANGUAGES, recommended_languages=['fa', 'en'], supports_auto_language=True,
        how_it_works='Whisper predicts text and timestamps from each speaker’s audio. '
                     'Task is transcribe, so the original language is retained.',
        limitations=['Not every language is supported; accuracy varies by language and recording.',
                     'Separation cross-talk can put another speaker’s words into the transcript.',
                     'First run downloads weights; silence and music can cause hallucinations.'])

    def load(self, language=None):
        if self._loaded:
            return
        import ctranslate2
        from faster_whisper import WhisperModel
        self._model_id = os.environ.get('STT_WHISPER_MODEL', 'small').strip() or 'small'
        # GPU visibility alone does not imply cuBLAS/cuDNN are installed (this
        # is common on Windows). CPU INT8 is the reliable out-of-box choice.
        device = os.environ.get('STT_DEVICE', 'cpu').strip()
        if device == 'auto':
            device = 'cuda' if ctranslate2.get_cuda_device_count() else 'cpu'
        compute = os.environ.get('STT_COMPUTE_TYPE') or ('int8_float16' if device == 'cuda' else 'int8')
        self._model = WhisperModel(self._model_id, device=device, compute_type=compute,
                                  download_root=str(PRETRAINED_DIR / 'stt/faster_whisper'),
                                  cpu_threads=max(1, int(os.environ.get('DS_THREADS', min(4, os.cpu_count() or 1)))))
        self._loaded = True

    def _recognize(self, samples, language):
        if language and language not in self._model.supported_languages:
            raise ValueError('The selected Whisper checkpoint does not support ' + language + '; avoid English-only .en models')
        try:
            segments, info = self._model.transcribe(samples, language=language, task='transcribe',
                beam_size=5, temperature=0., condition_on_previous_text=False, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200))
            rows = [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segments if s.text.strip()]
        except RuntimeError as exc:
            if any(token in str(exc).lower() for token in ('cublas', 'cudnn', 'cuda')):
                raise AppError('Whisper GPU inference failed: ' + str(exc),
                    fix='Set STT_DEVICE=cpu and restart the server, or install the CUDA/cuDNN libraries required by CTranslate2.') from exc
            raise
        return {'text': ' '.join(row['text'] for row in rows), 'language': info.language, 'segments': rows}
