# Contributing

Read the [main README](README.md), [fresh-clone setup](docs/SETUP.md) and
[verification guide](tests/README.md). The suggested everyday audio stack is
DeepFilterNet plus built-in clustering; preserve all three stages' selectable
algorithms and documented limitations.

## Thanks, credit and sharing improvements

Everyone is welcome to use and improve this project under its [MIT License](LICENSE).
Keep Nima Nadgaran's copyright notice and the license with copies or substantial
portions you distribute. Additional acknowledgement in your project's README is
appreciated, but is not an extra license requirement.

If you change something and make it better, you have my thanks! Open a pull
request, or email [thisisnima.nd1384@gmail.com](mailto:thisisnima.nd1384@gmail.com)
to let me know. Include a link to your changes, a short explanation and any
relevant test results so I can consider updating this project. Please do not
send credentials, private recordings or model weights by email.

Sharing improvements and contacting me are voluntary; you do not need to email
me or seek individual permission to use the project. Contributions submitted
for inclusion in this repository should be offered under the same MIT License,
and you must have the right to share them. Preserve third-party license notices.

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
