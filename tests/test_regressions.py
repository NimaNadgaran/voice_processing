"""Regression coverage for API isolation, validation, and long-running jobs."""
import asyncio
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from src.api import server
from src.api.jobs import Job, JobManager, MAX_EVENTS
from src.core.utils import contained_path
from src.pipeline import PipelineOptions, resolve_path
from src.pipeline.runner import source_stem


@pytest.mark.parametrize("path_id", ["../escape", "a/b", "a\\b", "C:\\escape", "CON", "x."])
def test_custom_paths_cannot_escape(path_id):
    with pytest.raises(ValueError):
        resolve_path({"id": path_id, "denoiser": "none", "separator": "none"})


def test_uploads_are_unique_and_keep_source_name(tmp_path):
    manager = JobManager()
    try:
        with patch("src.api.jobs.RAW_DIR", tmp_path), patch("src.api.jobs.time.strftime", return_value="20260914-120000"):
            first = manager.upload_target("meeting.wav")
            second = manager.upload_target("meeting.wav")
        assert first != second
        assert source_stem(first.name) == "meeting"
    finally:
        manager.shutdown()


def test_event_cursor_advances_after_retention_limit():
    job = Job("test", "x", "x", [])
    for i in range(MAX_EVENTS + 5):
        job.add_event({"message": str(i)})
    snapshot = job.snapshot(since=MAX_EVENTS)
    assert snapshot["next_seq"] == MAX_EVENTS + 5
    assert [e["seq"] for e in snapshot["events"]] == list(range(MAX_EVENTS, MAX_EVENTS + 5))
    assert job.snapshot(since=snapshot["next_seq"])["events"] == []


def test_file_guard_rejects_sibling_prefix(tmp_path):
    with pytest.raises(ValueError):
        contained_path(tmp_path / "job_a", "../job_ab/secret.wav")
    assert contained_path(tmp_path, "job_a/file.wav") == tmp_path / "job_a/file.wav"


@pytest.mark.parametrize("options", [{"waveform_points": 0}, {"max_speakers": -2}, {"num_speakers": -1}, {"count_on": "bad"}, [1], {"unknown": 1}])
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        PipelineOptions.from_dict(options)


@pytest.mark.parametrize("paths,options", [
    ('{}', '{}'), ('[]', '[1]'), ('[]', '{"waveform_points":0}'),
    ('[{"id":"../escape"}]', '{}'), ('["path1","path1"]', '{}'),
    ('[{"denoiser":"missing"}]', '{}'),
])
def test_invalid_requests_never_save_upload(paths, options):
    upload = UploadFile(filename="test.wav", file=io.BytesIO(b"audio"))
    with patch.object(server.MANAGER, "upload_target") as save:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(server.create_job(upload, paths, options))
        assert exc.value.status_code == 400
        save.assert_not_called()


def test_download_and_report_reject_traversal(tmp_path):
    (tmp_path / "job_a").mkdir()
    (tmp_path / "job_ab").mkdir()
    (tmp_path / "job_ab" / "secret.wav").write_bytes(b"secret")
    with patch.object(server, "OUT_DIR", tmp_path):
        with pytest.raises(HTTPException) as exc:
            server.get_file("job_a", "../job_ab/secret.wav")
        assert exc.value.status_code == 400
        with pytest.raises(HTTPException) as exc:
            server.job_report("../escape")
        assert exc.value.status_code == 400


def test_chunking_preserves_short_final_block():
    from src.separation.chunking import chunked_separate
    x = np.ones(203, dtype=np.float32)
    outputs = chunked_separate(lambda block: [block], x, 100, chunk_s=1, overlap_s=0.1, max_direct_s=0)
    np.testing.assert_allclose(outputs[0], x, atol=1e-5)


def test_pyannote_accepts_v4_result_wrapper():
    from types import SimpleNamespace
    from src.core.types import AudioBuffer
    from src.separation.methods.pyannote_diarize import PyannoteDiarizationSeparator
    backend = PyannoteDiarizationSeparator()
    annotation = SimpleNamespace(itertracks=lambda **kw: [(SimpleNamespace(start=0., end=1.), None, "speaker")])
    backend._model = lambda *args, **kw: SimpleNamespace(speaker_diarization=annotation)
    sources, info = backend._separate(AudioBuffer(np.ones(16000, np.float32), 16000), 1)
    assert len(sources) == 1
    assert info["metrics"]["detected_speakers"] == 1


def test_training_partitions_are_disjoint_and_reproducible():
    from src.core.training_split import held_out
    train, val = held_out(range(10), minimum=2)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(10))
    assert (train, val) == held_out(range(10), minimum=2)
    with pytest.raises(ValueError):
        held_out(range(3), minimum=2)


def test_training_dataloaders_hold_out_sources(tmp_path):
    from src.core.audio_io import save_audio
    from src.core.types import AudioBuffer
    from src.denoising.training.dataset import build_dataloaders as den_loaders
    from src.separation.training.dataset import build_dataloaders as sep_loaders
    for i in range(6):
        save_audio(tmp_path / str(i) / 'speech.wav', AudioBuffer(np.random.default_rng(i).normal(0, 0.1, 8000).astype(np.float32), 8000))
    cfg = dict(clean_dir=str(tmp_path), sample_rate=8000, segment_seconds=.5, steps_per_epoch=1, batch_size=1, val_items=1)
    train, val = den_loaders(cfg)
    assert not set(train.dataset.clean_files) & set(val.dataset.clean_files)
    first = train.dataset[0][0].numpy().copy()
    train.dataset.epoch = 1
    assert not np.array_equal(first, train.dataset[0][0].numpy())
    train, val = sep_loaders(cfg)
    assert not set(train.dataset.speaker_ids) & set(val.dataset.speaker_ids)
    assert next(iter(train))[1].shape[1] == 2
