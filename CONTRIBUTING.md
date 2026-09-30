# Contributing

Read the [main README](README.md), [fresh-clone setup](docs/SETUP.md) and
[verification guide](tests/README.md). The suggested everyday audio stack is
DeepFilterNet plus built-in clustering; preserve all three stages' selectable
algorithms and documented limitations.

## Backend changes

Methods register through `src/core/registry.py` and expose `MethodInfo` metadata.
Keep heavy optional imports/model loading inside runtime methods, so listing
methods does not require installing/downloading every neural backend. Missing
dependencies must produce a useful hint without breaking unrelated algorithms.

Preserve original audio sample rate, length/timeline, speaker labels and export
identity. Speaker counting and diarization are not acoustic source separation.
New recognizers must declare supported languages, distinguish recognition from
translation, bound long inputs, and preserve audio when text recognition fails.
Escape transcript/model/error content in frontend HTML and keep Unicode/RTL
text and UTF-8 downloads intact.

Add targeted regression tests and run checks appropriate to the risk. Use held-out
references for quality claims; distinguish mocked adapter tests, actual model
inference, exported file checks and subjective listening. Record missing model
access/dependencies explicitly rather than calling a backend verified.

## Data, credentials and publication

Do not commit personal recordings, datasets, checkpoints, cache directories,
generated reports, environments or credentials. The bundled synthetic frontend
demo is the intentional media exception. Check `.gitignore`, staged filenames
and [the upload checklist](docs/GITHUB.md) before committing.

Never put real tokens in example configs, remote URLs, issue descriptions or
screenshots. Redact logs/reports and confirm permission before sharing voice
recordings. Do not start full training or send audio to a hosted backend as a
side effect of an ordinary unit test.

Keep Markdown setup/API/training instructions consistent with the actual code.
Changing a recommendation in documentation does not silently change preset
algorithms. Update the stage guide and validation summary when behavior or
measured results change.
