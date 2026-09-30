"""Register lightweight declarations, without loading any speech models."""
from importlib import import_module

IMPORT_ERRORS = {}
for name in ('faster_whisper', 'whisper_transformers', 'persian_wav2vec2', 'vosk'):
    try:
        import_module('.' + name, __name__)
    except Exception as exc:
        IMPORT_ERRORS[name] = str(exc)
