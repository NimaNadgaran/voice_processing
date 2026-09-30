"""Bounded speech-to-text on each speaker's own track, preserving source time."""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..core.audio_io import resample
from ..core.errors import AppError, describe_exception, log_exception
from ..core.types import AudioBuffer, MethodInfo
from ..core.utils import missing_modules
from .languages import normalize_language


@dataclass
class TranscriptResult:
    method: str
    status: str = 'ok'
    text: str = ''
    language: str = ''
    segments: list = field(default_factory=list)
    elapsed: float = 0.
    model_id: str = ''
    metrics: dict = field(default_factory=dict)
    error: str = ''
    error_fix: str = ''

    def to_dict(self):
        return asdict(self)


def speech_windows(audio, regions=None, max_seconds=28., padding=.18):
    """Bounded, non-overlapping windows; use quiet boundaries when splitting turns.

    Existing turn gaps are not concatenated: returned offsets are always on the
    original speaker timeline. An explicit empty region list means no speech.
    """
    if max_seconds <= 0 or audio.sr <= 0:
        raise ValueError('positive window duration and sample rate required')
    regions = [[0., audio.duration]] if regions is None else regions
    merged = []
    for pair in sorted(regions):
        start, end = map(float, pair)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError('speech turns must have finite timestamps')
        start, end = max(0., start - padding), min(audio.duration, end + padding)
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + .5:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    limit = max(1, int(max_seconds * audio.sr))
    for start, end in merged:
        cursor, stop = int(start * audio.sr), min(audio.n_samples, int(end * audio.sr))
        while cursor < stop:
            boundary = min(cursor + limit, stop)
            if boundary < stop and limit >= audio.sr * 4:
                # A quiet 40 ms cut within the final two seconds loses less
                # speech context than splitting every fixed 28 s mid-word.
                hop = max(1, int(.04 * audio.sr))
                candidates = np.arange(boundary - 2 * audio.sr, boundary, hop)
                energies = [float(np.mean(audio.samples[p:p + hop] ** 2)) for p in candidates]
                boundary = int(candidates[int(np.argmin(energies))] + hop)
            samples = audio.samples[cursor:boundary]
            if len(samples) and float(np.max(np.abs(samples))) > 1e-6:
                yield cursor / audio.sr, samples
            cursor = boundary


class BaseTranscriber:
    info = MethodInfo('base', 'Base transcriber', 'transcribe', 'deep-pretrained', '')
    fixed_language = None

    def __init__(self):
        self._run_lock = threading.RLock()
        self._loaded = False
        self._model = None
        self._model_id = ''

    def check_available(self):
        missing = missing_modules(self.info.pip)
        return (False, 'missing python package(s): ' + ', '.join(missing)) if missing else (True, '')

    def describe(self):
        info = copy.deepcopy(self.info)
        try:
            info.available, info.unavailable_reason = self.check_available()
        except Exception as exc:
            info.available, info.unavailable_reason = False, str(exc)
        return info

    def validate_language(self, language):
        language = normalize_language(language)
        if language is None and self.fixed_language:
            return self.fixed_language
        if language is None and not self.info.supports_auto_language:
            raise ValueError(self.info.name + ' needs an explicit language')
        if language and language not in self.info.supported_languages:
            raise ValueError(self.info.name + ' does not support ' + language)
        return language

    def load(self, language=None):
        raise NotImplementedError

    def _recognize(self, samples, language):
        """Return text/segments with timestamps relative to one 16 kHz window."""
        raise NotImplementedError

    def run(self, audio, language=None, regions=None, progress=None):
        with self._run_lock:
            started = time.perf_counter()
            language = self.validate_language(language)
            if audio.sr <= 0 or not np.isfinite(audio.samples).all():
                raise ValueError('speech-to-text requires finite audio and a positive sample rate')
            work = resample(audio, 16000) if audio.sr != 16000 else audio
            if not work.n_samples or work.peak() <= 1e-6 or regions == []:
                return TranscriptResult(self.info.key, 'empty', language=language or '',
                                        elapsed=time.perf_counter() - started)
            ok, reason = self.check_available()
            if not ok:
                raise AppError(self.info.name + ' is not ready: ' + reason,
                               fix=self.info.install_hint, kind='missing-dependency')
            if progress:
                progress(0., 'Loading ' + self.info.name + ' (first run may download weights)')
            self.load(language)
            segments, texts, detected, warnings = [], [], [], []
            for offset, samples in speech_windows(work, regions):
                if progress:
                    progress(min(.95, offset / max(work.duration, .001)),
                             'Transcribing at %.1fs with %s' % (offset, self.info.name))
                data = self._recognize(np.ascontiguousarray(samples, dtype=np.float32), language)
                text = str(data.get('text', '')).strip()
                if text:
                    texts.append(text)
                    # A quality warning, not proof of a recognition error:
                    # genuine stuttering/repeated speech is also possible.
                    words = text.casefold().split()
                    if len(words) >= 40 and len(set(words)) / len(words) <= .2:
                        warnings.append('Highly repetitive recognition; review against the audio or try a larger model.')
                if data.get('language'):
                    detected.append(data['language'])
                for segment in data.get('segments') or ([dict(start=0., end=len(samples) / 16000., text=text)] if text else []):
                    start, end = float(segment.get('start', 0)), float(segment.get('end', len(samples) / 16000))
                    if not math.isfinite(start) or not math.isfinite(end):
                        raise RuntimeError('ASR returned non-finite timestamps')
                    start, end = max(0., start), min(len(samples) / 16000., end)
                    if end > start and str(segment.get('text', '')).strip():
                        segments.append({**segment, 'start': round(offset + start, 3),
                                         'end': round(offset + end, 3), 'text': str(segment['text']).strip()})
            languages = list(dict.fromkeys(detected))
            result = TranscriptResult(self.info.key, 'ok' if texts else 'empty',
                text='\n'.join(texts), language=language or (languages[0] if len(languages) == 1 else 'mixed' if languages else ''),
                segments=segments, elapsed=time.perf_counter() - started, model_id=self._model_id,
                metrics={'languages': languages, 'timestamp_precision': 'model-dependent', 'off_device': False,
                         'warnings': list(dict.fromkeys(warnings))})
            if progress:
                progress(1., 'Transcript ready')
            return result

    def safe_run(self, *args, **kwargs):
        try:
            return self.run(*args, **kwargs)
        except Exception as exc:
            log_exception('transcribe:' + self.info.key, exc)
            info = describe_exception(exc, 'speech-to-text')
            return TranscriptResult(self.info.key, 'failed', error=info['message'], error_fix=info['fix'])

    def unload(self):
        with self._run_lock:
            self._model = None
            self._loaded = False


def write_transcript(path, result):
    """Plain UTF-8 text; no ASCII mangling of Persian or other scripts."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.text.rstrip() + ('\n' if result.text else ''), encoding='utf-8', newline='\n')
    return path
