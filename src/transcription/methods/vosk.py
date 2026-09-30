"""Lightweight streaming Kaldi/Vosk recognition with separate language models."""
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from ...core.registry import register_transcriber
from ...core.types import MethodInfo
from ...core.utils import PRETRAINED_DIR
from ..base import BaseTranscriber

MODEL_NAMES = {'en': 'vosk-model-small-en-us-0.15', 'fa': 'vosk-model-small-fa-0.42'}


def model_directory(language):
    override = os.environ.get('STT_VOSK_MODEL_DIR', '').strip()
    if override:
        declared = os.environ.get('STT_VOSK_LANGUAGE', '').strip().lower()
        if declared != language:
            raise ValueError('STT_VOSK_LANGUAGE must match the selected language for STT_VOSK_MODEL_DIR')
        folder = Path(override)
        if not (folder / 'am/final.mdl').is_file():
            raise ValueError('STT_VOSK_MODEL_DIR is not an extracted Vosk model')
        return folder
    name = MODEL_NAMES[language]
    cache = PRETRAINED_DIR / 'stt/vosk'
    target = cache / name
    if (target / 'am/final.mdl').is_file():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    # Do not publish an incomplete model directory if download/extraction fails.
    with tempfile.TemporaryDirectory(prefix='download_', dir=cache) as temp:
        root = Path(temp)
        archive = root / 'model.zip'
        with urllib.request.urlopen('https://alphacephei.com/vosk/models/' + name + '.zip', timeout=90) as source:
            with archive.open('wb') as output:
                shutil.copyfileobj(source, output)
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                resolved = (root / item.filename).resolve()
                if not resolved.is_relative_to(root.resolve()) or item.filename.split('/')[0] != name:
                    raise ValueError('Unsafe path in Vosk model archive')
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Links are not allowed in model archives')
            zf.extractall(root)
        unpacked = root / name
        if not (unpacked / 'am/final.mdl').is_file():
            raise ValueError('Downloaded Vosk model is incomplete')
        if target.exists():
            raise RuntimeError('Incomplete Vosk cache exists: ' + str(target))
        shutil.move(str(unpacked), str(target))
    return target


@register_transcriber
class VoskTranscriber(BaseTranscriber):
    info = MethodInfo(
        'vosk', 'Vosk — lightweight English / Persian', 'transcribe', 'deep-pretrained',
        'Low-memory CPU speech recognition with separate English and Persian models. '
        'Choose the language explicitly; this engine does not detect it.', speed='realtime', quality=3,
        pip=['vosk'], install_hint='pip install vosk', supported_languages=['en', 'fa'],
        notes='First selected language downloads its small model (~40–53 MB). '
              'STT_VOSK_MODEL_DIR + STT_VOSK_LANGUAGE can select an extracted model.',
        recommended_languages=['en', 'fa'],
        limitations=['Requires an explicit language and matching model.',
                     'Small models trade accuracy and punctuation for speed.'])

    def __init__(self):
        super().__init__()
        self._models = {}

    def load(self, language=None):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        if language not in self._models:
            self._models[language] = Model(str(model_directory(language)))
        self._model = self._models[language]
        self._model_id = os.environ.get('STT_VOSK_MODEL_DIR', '').strip() or MODEL_NAMES[language]
        self._loaded = True

    def _recognize(self, samples, language):
        from vosk import KaldiRecognizer
        recognizer = KaldiRecognizer(self._model, 16000)
        recognizer.SetWords(True)
        pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
        results = []
        for start in range(0, len(pcm), 8000):
            if recognizer.AcceptWaveform(pcm[start:start + 8000]):
                results.append(json.loads(recognizer.Result()))
        results.append(json.loads(recognizer.FinalResult()))
        texts, segments = [], []
        for item in results:
            text = item.get('text', '').strip()
            if not text:
                continue
            texts.append(text)
            words = item.get('result', [])
            segments.append(dict(start=words[0]['start'] if words else 0.,
                                 end=words[-1]['end'] if words else len(samples) / 16000,
                                 text=text))
        return dict(text=' '.join(texts), language=language, segments=segments)

    def unload(self):
        with self._run_lock:
            self._models.clear()
            super().unload()
