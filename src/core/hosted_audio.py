"""Hosted audio requests with isolated payloads and actionable service errors."""
import io
import os
import time

from .errors import AppError

API_ROOT = 'https://router.huggingface.co/hf-inference/models/'


def request_audio(audio, model_id, token, stage, timeout=180):
    import requests
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio.samples, audio.sr, format='WAV', subtype='PCM_16')
    url = os.environ.get(f'HF_{stage.upper()}_URL', '').strip() or API_ROOT + model_id
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'audio/wav'}
    for attempt in range(4):
        response = requests.post(url, headers=headers, data=buffer.getvalue(), timeout=timeout)
        if response.status_code == 200:
            return response, attempt + 1
        if response.status_code in (401, 403):
            raise AppError('The hosted audio service rejected the token or model access.',
                           fix='Check HF_TOKEN and access to the configured endpoint.')
        if response.status_code in (404, 410):
            raise AppError('The configured model has no audio-to-audio endpoint at this address.',
                           fix=f'Set HF_{stage.upper()}_URL to a deployed audio-to-audio endpoint, '
                               'or choose a local method. A model on the Hub is not necessarily hosted.')
        if response.status_code not in (429, 503) or attempt == 3:
            raise AppError(f'Hosted audio inference failed (HTTP {response.status_code}).',
                           fix='Check the endpoint status, model support, and account quota.')
        wait = 8 * (attempt + 1)
        try:
            wait = float(response.headers.get('Retry-After') or response.json().get('estimated_time', wait))
        except (ValueError, TypeError, AttributeError):
            pass
        time.sleep(max(0., min(wait, 30.)))
    raise RuntimeError('Hosted audio service did not return a result')
