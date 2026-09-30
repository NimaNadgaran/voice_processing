"""Alternative local Whisper inference through Hugging Face Transformers."""
import os
import copy

from ...core.registry import register_transcriber
from ...core.types import MethodInfo
from ...core.utils import PRETRAINED_DIR
from ..base import BaseTranscriber
from ..languages import WHISPER_LANGUAGES


@register_transcriber
class TransformersWhisperTranscriber(BaseTranscriber):
    info = MethodInfo(
        'whisper_transformers', 'Whisper (Transformers) — multilingual', 'transcribe', 'deep-pretrained',
        'Alternative PyTorch Whisper inference with automatic or pinned language, including Persian.',
        speed='medium', quality=4, pip=['torch', 'transformers'],
        install_hint='pip install torch transformers',
        notes='Default openai/whisper-small. Set STT_HF_WHISPER_MODEL to a compatible Whisper model or local directory.',
        supported_languages=WHISPER_LANGUAGES, supports_auto_language=True,
        recommended_languages=['fa', 'en'],
        limitations=['First run downloads weights.', 'Uses more CPU memory than the INT8 backend.',
                     'Different languages and dialects have different accuracy.'])

    def load(self, language=None):
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        self._device = os.environ.get('STT_DEVICE', 'auto')
        if self._device == 'auto':
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_id = os.environ.get('STT_HF_WHISPER_MODEL', 'openai/whisper-small')
        cache = str(PRETRAINED_DIR / 'stt/huggingface')
        self._processor = AutoProcessor.from_pretrained(self._model_id, cache_dir=cache, trust_remote_code=False)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self._model_id, cache_dir=cache,
                                                              trust_remote_code=False).to(self._device).eval()
        self._loaded = True

    def _recognize(self, samples, language):
        import torch
        inputs = self._processor(samples, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        kwargs = dict(task='transcribe', language=language, max_new_tokens=440, num_beams=5,
                      do_sample=False, no_repeat_ngram_size=6, return_dict_in_generate=True,
                      return_timestamps=False)
        # A fine-tuned model may carry a pinned language in its generation
        # configuration. Passing language=None alone does not reset it.
        config = getattr(self._model, 'generation_config', None)
        english_only = config is not None and getattr(config, 'is_multilingual', True) is False
        if english_only:
            if language not in (None, 'en'):
                raise ValueError('The selected Whisper checkpoint only supports English; choose a multilingual model for ' + language)
            kwargs.update(task=None, language=None)
        if config is not None:
            config = copy.deepcopy(config)
            config.language = None if english_only else language
            config.forced_decoder_ids = None
            kwargs['generation_config'] = config
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **kwargs)
        # ModelOutput includes the decoder prefix and its detected language;
        # the default plain tensor omits this prefix in newer Transformers.
        tokens = generated.sequences if hasattr(generated, 'sequences') else generated
        text = self._processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        import re
        from ..languages import LANGUAGES
        raw = self._processor.batch_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        detected = next((code for code in re.findall(r'<\|([a-z]{2,3})\|>', raw) if code in LANGUAGES), '')
        if english_only:
            detected = 'en'
        # HF short-form generation yields text, not forced word alignment.
        return {'text': text, 'language': language or detected,
                'segments': [{'start': 0., 'end': len(samples) / 16000, 'text': text}] if text else []}
