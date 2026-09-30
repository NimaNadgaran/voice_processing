"""Measured training ETAs, including validation, without changing model state."""
from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path


@contextmanager
def preserve_training_state(model, optimiser, scaler):
    import torch
    weights = copy.deepcopy(model.state_dict())
    optim = copy.deepcopy(optimiser.state_dict())
    scale = copy.deepcopy(scaler.state_dict())
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode = model.training
    try:
        yield
    finally:
        model.load_state_dict(weights)
        optimiser.load_state_dict(optim)
        scaler.load_state_dict(scale)
        optimiser.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(mode)


def time_batches(step, loader, device, n_steps=8):
    import torch
    times = []
    iterator = iter(loader)
    for _ in range(min(n_steps, len(loader))):
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch = next(iterator)
        step(batch)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    if not times:
        raise ValueError("Training loader is empty; reduce batch size or supply more data")
    stable = times[2:] or times
    return sum(stable) / len(stable)


def measured_estimate(model, optimiser, scaler, train_timing, validation, val_loader, cfg, start_epoch=0, path=None):
    import torch
    with preserve_training_state(model, optimiser, scaler):
        per_step = train_timing()
        if isinstance(per_step, dict):
            per_step = per_step['seconds_per_step']
        batches = min(3, len(val_loader))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        validation(islice(val_loader, batches))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        val_seconds = (time.perf_counter() - start) * len(val_loader) / max(batches, 1)
    epoch_seconds = per_step * cfg['steps_per_epoch'] + val_seconds
    remaining = max(0, cfg['epochs'] - start_epoch) * epoch_seconds
    estimate = dict(seconds_per_step=per_step, validation_seconds_per_epoch=val_seconds,
                    seconds_per_epoch=epoch_seconds, remaining_seconds=remaining,
                    epochs=cfg['epochs'], start_epoch=start_epoch,
                    estimated_finish=finish_at(remaining), config=cfg)
    if path:
        write_estimate(path, estimate)
    print(f"Measured {per_step:.3f} s/train step; {val_seconds:.1f} s validation/epoch.")
    print(f"Training ETA: {duration(remaining)}; expected finish {estimate['estimated_finish']}", flush=True)
    return estimate


def write_estimate(path, estimate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate, indent=2), encoding='utf-8')


def finish_at(seconds):
    return (datetime.now().astimezone() + timedelta(seconds=seconds)).isoformat(timespec='seconds')


def duration(seconds):
    minutes, sec = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02}h {minutes:02}m {sec:02}s" if days else f"{hours:02}h {minutes:02}m {sec:02}s"


def report_remaining(elapsed, completed_epochs, remaining_epochs):
    remaining = elapsed / max(1, completed_epochs) * remaining_epochs
    print(f"Updated ETA: {duration(remaining)} remaining; finish {finish_at(remaining)}", flush=True)
    return remaining
