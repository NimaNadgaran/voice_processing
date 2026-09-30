"""Headless Chrome regression: selector language support, safe text and downloads.

Requires pip install playwright and an existing Chrome/Edge installation.
    python scripts/check_transcription_ui.py --url http://127.0.0.1:9139
No browser download, no recording upload, and no external audio transmission.
"""
import argparse
import io
import sys
import uuid
import zipfile
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:9139')
    parser.add_argument('--run-pipeline', type=Path, help='Also upload and transcribe a real two-speaker WAV')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.transcription.base import write_transcript, TranscriptResult
    folder = root / 'data/outputs/stt_ui_verification'
    folder.mkdir(parents=True, exist_ok=True)
    job_id = 'stt_browser_' + uuid.uuid4().hex[:6]
    # Use the actual guarded file endpoint, not a mocked download response.
    write_transcript(root / 'data/outputs' / job_id / 'path1/speaker01.txt',
                     TranscriptResult('test', text='سلام دنیا\nمتن فارسی'))
    report = {'job_id': job_id, 'elapsed_human': '1 s', 'input': {'duration': 3},
        'paths': [{'id': 'path1', 'name': 'Browser transcript check', 'status': 'ok',
            'denoiser': 'none', 'separator': 'none', 'timings': {'total': 1},
            'transcription': {'status': 'ok', 'method': 'faster_whisper'},
            'tracks': [{'label': 'Speaker 1', 'file': 'speaker01.wav', 'segments': [[0, 3]],
                'transcription': {'status': 'ok', 'method': 'faster_whisper', 'language': 'fa',
                    'metrics': {'warnings': ['Review repetitive text against the audio.']},
                    'text': 'سلام دنیا\nمتن فارسی <script>alert(1)</script>', 'file': 'speaker01.txt'}}]}]}
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': 1360, 'height': 1000}, accept_downloads=True)
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        page.goto(args.url)
        page.wait_for_selector('#opt-language option[value="fa"]', state='attached')
        assert page.locator('#opt-transcriber').input_value() == 'auto'
        assert page.locator('#transcriber-list .method').count() == 4
        page.select_option('#opt-transcriber', 'persian_wav2vec2')
        # Native <option> disabledness is best checked directly: Playwright's
        # generic control-state helper does not cover it consistently in Chrome.
        assert page.locator('#opt-language option[value="en"]').evaluate('(option) => option.disabled')
        page.select_option('#opt-transcriber', 'faster_whisper')
        page.select_option('#opt-language', 'fa')
        page.select_option('.crow-stt', 'vosk')
        assert page.locator('.crow-stt').input_value() == 'vosk'
        page.check('.crow-on')
        assert page.locator('#opt-language option[value="auto"]').evaluate('(option) => option.disabled')
        page.select_option('#opt-transcriber', 'none')
        assert page.locator('#opt-language').is_enabled()  # explicit row still needs language
        page.select_option('#opt-transcriber', 'faster_whisper')
        page.evaluate('report => renderResultCard(report.paths[0], report)', report)
        text = page.locator('.transcript-text')
        assert 'سلام دنیا' in text.inner_text()
        assert text.get_attribute('dir') == 'auto'
        assert page.locator('.transcript-box script').count() == 0
        assert 'Review repetitive' in page.locator('.transcript-warning').inner_text()
        with page.expect_download() as capture:
            page.get_by_role('link', name='Download text file', exact=True).click()
        download = capture.value
        download.save_as(folder / download.suggested_filename)
        assert (folder / 'speaker01.txt').read_text(encoding='utf-8').startswith('سلام دنیا')
        page.locator('.result-card').screenshot(path=str(folder / 'desktop.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(folder / 'mobile_page.png'), full_page=True)
        overflow = page.evaluate('''() => [...document.querySelectorAll('body *')]
            .filter(el => el.getBoundingClientRect().right > innerWidth + 1)
            .map(el => ({tag: el.tagName, cls: el.className,
                        right: el.getBoundingClientRect().right})).slice(0, 25)''')
        if overflow:
            print('Mobile overflow:', overflow)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.locator('.transcript-box').screenshot(path=str(folder / 'mobile_transcript.png'))
        if args.run_pipeline:
            page.set_viewport_size({'width': 1360, 'height': 1000})
            page.evaluate("state.selected = new Set(['path1']); state.custom = []; renderPaths(); renderCustomRows(); updateRunState()")
            page.select_option('#opt-transcriber', 'faster_whisper')
            page.select_option('#opt-language', 'en')
            page.select_option('#opt-speakers', '2')
            page.set_input_files('#file-input', str(args.run_pipeline.resolve()))
            page.wait_for_function("!document.querySelector('#run-btn').disabled")
            page.click('#run-btn')
            page.wait_for_function("document.querySelector('#progress-sub').textContent.startsWith('finished in ')", timeout=180000)
            links = page.get_by_role('link', name='Download text file', exact=True)
            assert links.count() == 2, page.locator('#results').inner_text()
            for index in range(links.count()):
                with page.expect_download() as capture:
                    links.nth(index).click()
                saved = folder / capture.value.suggested_filename
                capture.value.save_as(saved)
                assert saved.read_text(encoding='utf-8').strip()
            with page.expect_download() as capture:
                page.get_by_role('link', name='Download this path (.zip)', exact=True).click()
            saved_zip = folder / capture.value.suggested_filename
            capture.value.save_as(saved_zip)
            with zipfile.ZipFile(io.BytesIO(saved_zip.read_bytes())) as archive:
                assert len([name for name in archive.namelist() if name.endswith('.txt')]) == 2
            page.locator('.result-card').screenshot(path=str(folder / 'real_pipeline.png'))
            print('Real browser pipeline passed: upload, denoise, two speakers, two nonempty text downloads and ZIP.')
        assert not errors, errors
        browser.close()
    print('Chrome passed: four backends, Persian restriction, per-path selection, safe RTL preview, UTF-8 download and mobile width.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
