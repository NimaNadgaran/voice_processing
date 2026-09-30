# GitHub upload checklist

Upload reproducible source files, not your environment, recordings or model
cache. This guide does not itself stage, commit or push anything.

## Included source

- `.gitignore`, all project Markdown guides and root launchers (`run.py`,
  `main.py`, `run.ps1`, training `setup.py`).
- `requirements*.txt`, `src/` Python modules and training YAML configurations.
- `frontend/` HTML/CSS/JS and its small synthetic `demo_conversation.wav` asset.
- `scripts/` development/benchmark/download utilities and `tests/`.
- `data/README.md`, `models/README.md`, and their existing `.gitkeep` markers.

The demo is intentionally allowed despite the general rule excluding personal
audio. Configs and requirements are intentionally allowed; do not blanket-ignore
all JSON, YAML, CSV or TXT files.

## Kept local by `.gitignore`

Environments (`.venv*`, `venv*`, `env`), downloaded Python runtimes / UV cache,
bytecode, test/linter/coverage caches, local editor/agent settings, `.env` files,
tokens/keys, training datasets, uploads, outputs, caches, sample recordings,
model weights, checkpoint files, archives, logs and `runs/`.

Only the explicit data/model documentation and empty-directory markers are
re-included. Downloaded third-party READMEs inside caches do not become source
just because they end in `.md`. `.env.example` / `.env.template` can be tracked
if they contain placeholders only; environment files are not auto-loaded by
the app and must not contain real credentials in Git.

Ignored files remain on disk. A fresh clone creates local runtime directories
and downloads/trains models; see [SETUP.md](SETUP.md) and [TRAINING.md](TRAINING.md).
Share trained weights separately only when you have permission and have checked
their licensing. A model cache is not required to upload the application code.

## Review before staging

```bash
git status --short
git diff --check
git diff --stat
git ls-files
git ls-files -ci --exclude-standard
git status --short --ignored
```

`git ls-files -ci --exclude-standard` identifies files already tracked that now
match ignore rules. `.gitignore` alone cannot stop Git publishing a tracked file.
For a specific unwanted local artifact, use `git rm --cached -- <exact-path>`;
this removes it from the index **without deleting the local copy**. Review the
result; do not run a broad cache-removal command across the repository.

The local package inventory `data/installed-before.json` belongs to runtime data,
not the published source. Review any new datasets, transcripts, logs, recordings
or secrets before staging. Ignore rules are a safeguard, not a secret scanner.
If credentials were published previously, ignoring them now does not remove
them from history; revoke/rotate exposed credentials through the provider.

Useful spot checks:

```bash
git check-ignore -v --no-index .venv/pyvenv.cfg data/outputs/example/report.json models/checkpoints/example.pt .env
git check-ignore -v --no-index frontend/assets/demo_conversation.wav data/raw/.gitkeep
```

With `-v`, a rule beginning with `!` is an inclusion exception, not an exclusion.
The bundled demo and `.gitkeep` markers should remain uploadable.

## Stage, inspect, commit, then push

After reviewing all current changes yourself:

```bash
git add .
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
git commit -m "Document setup, suggested algorithms and GitHub upload rules"
```

Confirm your remote and branch before pushing:

```bash
git remote -v
git branch --show-current
```

This checkout currently uses `origin` and `master`; if that still matches your
intended GitHub repository:

```bash
git push -u origin master
```

Do not put access tokens in remote URLs or screenshots. On another checkout,
substitute its actual branch and configured remote; do not rename a branch or
overwrite remote history merely to match this example. For a new remote use
your repository's URL and normal Git authentication; no force-push is needed for
the documentation/ignore updates.

## Final publication checks

Check the rendered main README and its relative links, confirm code/YAML/TXT
requirements and the demo are present, and verify that recordings, weights,
datasets and credentials are not in the proposed commit. Include the root
[MIT License](../LICENSE) and preserve Nima Nadgaran's copyright notice. Review
the separate licenses of dependencies, models and datasets before distributing
any third-party material; this project license does not replace their terms.
Verify fresh-clone instructions rather than relying on your local cached files.
GitHub is hosting the source, not the live Python application or training queue.
