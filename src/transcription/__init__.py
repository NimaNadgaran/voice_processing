"""Stage 3: local multilingual speech-to-text, independently callable."""
from pathlib import Path

from ..core.audio_io import load_audio
from ..core.errors import describe_exception, log_exception
from ..core.registry import get_transcriber, list_transcribers
from ..core.types import AudioBuffer
from .base import TranscriptResult, write_transcript
from .languages import normalize_language


def resolve_method(method='auto', language=None):
    language = normalize_language(language)
    if method != 'auto':
        return method
    # Prefer the efficient multilingual model; never silently choose Persian
    # for an unknown-language recording or send audio to a hosted service.
    keys = ['faster_whisper', 'whisper_transformers']
    if language == 'fa':
        keys += ['persian_wav2vec2']
    if language in ('fa', 'en'):
        keys += ['vosk']
    for key in keys:
        backend = get_transcriber(key)
        if backend.describe().available:
            return key
    raise RuntimeError('No installed speech-to-text backend supports this language. '
                       'Install: pip install -r requirements-stt.txt')


def transcribe(audio, method='auto', language=None, output_path=None,
               regions=None, progress=None, safe=True):
    if method == 'none':
        return TranscriptResult('none', 'disabled')
    try:
        if not isinstance(audio, AudioBuffer):
            audio = load_audio(Path(audio))
        resolved = resolve_method(method, language)
        backend = get_transcriber(resolved)
        result = backend.safe_run(audio, language, regions, progress) if safe else backend.run(audio, language, regions, progress)
        if output_path and result.status in ('ok', 'empty'):
            write_transcript(output_path, result)
        return result
    except Exception as exc:
        if not safe:
            raise
        log_exception('transcribe:' + method, exc)
        info = describe_exception(exc, 'speech-to-text')
        fix = info['fix'] or ('pip install -r requirements-stt.txt' if 'No installed speech-to-text' in str(exc) else '')
        return TranscriptResult(method, 'failed', error=info['message'], error_fix=fix)


__all__ = ['transcribe', 'list_transcribers', 'TranscriptResult']
