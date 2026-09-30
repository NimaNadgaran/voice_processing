"""Run actual local speech recognition with format and optional reference checks.

    python scripts/audit_transcription.py --input recording.wav --language fa
    python scripts/audit_transcription.py --methods faster_whisper,vosk --language en

The default input is one recorded LibriSpeech utterance and its known English
reference. Explicit recordings have no accuracy score unless --reference-text
is supplied. Reports distinguish failed, empty and successful recognition;
success is not an accuracy guarantee. Every successful result exports a TXT.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.audio_io import load_audio, save_audio
from src.core.types import AudioBuffer
from src.core.utils import write_json, validate_component
from src.transcription import transcribe


def error_rate(reference, hypothesis, characters=False):
    def normalized(text):
        text = text.lower().replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s]', '', text)
        return list(''.join(text.split())) if characters else text.split()
    ref, hyp = normalized(reference), normalized(hypothesis)
    costs = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, 1):
        previous, costs[0] = costs[0], i
        for j, right in enumerate(hyp, 1):
            old = costs[j]
            costs[j] = min(costs[j] + 1, costs[j - 1] + 1, previous + (left != right))
            previous = old
    return costs[-1] / max(1, len(ref))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--language', default='en')
    parser.add_argument('--methods', default='faster_whisper,vosk')
    parser.add_argument('--reference-text')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--label', default='english')
    args = parser.parse_args()
    if args.start < 0 or args.seconds <= 0:
        parser.error('--start must be non-negative and --seconds must be positive')
    methods = [value.strip() for value in args.methods.split(',') if value.strip()]
    if not methods:
        parser.error('--methods needs at least one engine')
    for method in methods:
        validate_component(method)
    validate_component(args.label)
    folder = ROOT / 'data/outputs' / ('stt_audit_' + args.label)
    folder.mkdir(parents=True, exist_ok=True)
    source = args.input
    reference = args.reference_text
    if source is None:
        corpus = ROOT / 'data/datasets/_download/LibriSpeech/dev-clean'
        source = next(corpus.rglob('*.flac'))
        transcript = source.parent / (source.parent.parent.name + '-' + source.parent.name + '.trans.txt')
        if transcript.exists():
            for line in transcript.read_text(encoding='utf-8').splitlines():
                key, text = line.split(' ', 1)
                if key == source.stem:
                    reference = text
                    break
    audio = load_audio(source)
    if args.input:
        audio = AudioBuffer(audio.samples[int(args.start * audio.sr):int((args.start + args.seconds) * audio.sr)], audio.sr)
    save_audio(folder / 'input.wav', audio)
    rows = []
    for method in methods:
        print('Recognizing with ' + method, flush=True)
        result = transcribe(audio, method=method, language=args.language,
                            output_path=folder / (method + '.txt'),
                            progress=lambda pct, message: print(message, flush=True))
        row = result.to_dict()
        if result.status in ('ok', 'empty') and reference:
            row.update(reference_text=reference, word_error_rate=error_rate(reference, result.text),
                       character_error_rate=error_rate(reference, result.text, True))
        rows.append(row)
        print(result.status + ': ' + ascii(result.text), flush=True)
        if result.error:
            print(result.error, flush=True)
        write_json(folder / 'report.json', dict(source=str(source), rows=rows))
    return 1 if any(row['status'] == 'failed' for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
