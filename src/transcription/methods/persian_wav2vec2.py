"""Persian-specialized CTC recognition, not a multilingual or translation model."""
import os

from ...core.registry import register_transcriber
from ...core.types import MethodInfo
from ...core.utils import PRETRAINED_DIR
from ..base import BaseTranscriber


@register_transcriber
class PersianWav2Vec2Transcriber(BaseTranscriber):
    fixed_language = 'fa'
    info = MethodInfo(
        'persian_wav2vec2', 'Wav2Vec2 XLSR — Persian specialist', 'transcribe', 'deep-pretrained',
        'CTC model fine-tuned specifically on Persian/Farsi Common Voice speech. '
        'An alternative to multilingual Whisper, not a claim of universally better Persian accuracy.',
        speed='medium', quality=3, pip=['torch', 'transformers'],
        install_hint='pip install torch transformers', supported_languages=['fa'], recommended_languages=['fa'],
        notes='jonatasgrosman/wav2vec2-large-xlsr-53-persian (~1.2 GB). Requires Persian input; no automatic language detection.',
        limitations=['Persian only; auto selects Persian because the model is language-specific.',
                     'Greedy CTC output may lack punctuation and needs testing on your dialect.',
                     'First run downloads approximately 1.2 GB.'])

    def load(self, language=None):
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        self._device = os.environ.get('STT_DEVICE', 'auto')
        if self._device == 'auto':
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_id = os.environ.get('STT_PERSIAN_MODEL', 'jonatasgrosman/wav2vec2-large-xlsr-53-persian')
        # Loading an older .bin checkpoint must not start an unsolicited
        # remote safetensors-conversion job or a second full weight download.
        os.environ.setdefault('DISABLE_SAFETENSORS_CONVERSION', '1')
        cache = str(PRETRAINED_DIR / 'stt/huggingface')
        self._processor = AutoProcessor.from_pretrained(self._model_id, cache_dir=cache, trust_remote_code=False)
        self._model = AutoModelForCTC.from_pretrained(self._model_id, cache_dir=cache,
                                                   trust_remote_code=False).to(self._device).eval()
        self._loaded = True

    def _recognize(self, samples, language):
        import torch
        # Encoder kernels require more than a handful of samples.
        import numpy as np
        samples = np.pad(samples, (0, max(0, 800 - len(samples))))
        inputs = self._processor(samples, sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.inference_mode():
            logits = self._model(**{key: value.to(self._device) for key, value in inputs.items()}).logits
        text = self._processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
        return {'text': text, 'language': 'fa'}
