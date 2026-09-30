"""Train the local SpectralUNet denoiser.

    python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml

Nothing in the web app ever calls this: training only starts when *you* run this
command.  The first 20 steps are timed and the script prints a measured ETA for
the whole run before it commits to it, so you never discover after 6 hours that
the config was too big for your machine.

Useful flags
------------
    --config PATH        yaml with everything below (CLI flags win)
    --preset tiny|base   size shortcut (tiny = laptop-CPU friendly)
    --epochs N           override
    --device cpu|cuda    override
    --estimate-only      build the model, time a few steps, print the ETA, exit
    --resume PATH        continue from a checkpoint
    --max-hours H        stop cleanly after H hours (checkpoint is still saved)

Outputs
-------
    models/checkpoints/denoise_unet_last.pt
    models/checkpoints/denoise_unet_best.pt      <- the app picks this one up
    runs/denoise_unet/...                          tensorboard logs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRESETS: Dict[str, Dict[str, Any]] = {
    # CPU-sized network at 16 kHz, preserving consonants above the tiny model's
    # 4 kHz cutoff. Used by setup.py on machines without CUDA.
    "compact": dict(sample_rate=16000, segment_seconds=2.0, n_fft=512, hop=128,
                    base_channels=16, depth=3, batch_size=8, steps_per_epoch=400, epochs=30),
    # laptop-CPU friendly: ~0.35 M params, 8 kHz, 1 s crops
    "tiny": dict(sample_rate=8000, segment_seconds=1.0, n_fft=256, hop=128,
                 base_channels=16, depth=3, batch_size=16, steps_per_epoch=400, epochs=30),
    # the default that the app expects: ~1.6 M params, 16 kHz, 2 s crops
    "base": dict(sample_rate=16000, segment_seconds=2.0, n_fft=512, hop=128,
                 base_channels=32, depth=4, batch_size=8, steps_per_epoch=1000, epochs=100),
    # for a real GPU and a real corpus
    "large": dict(sample_rate=16000, segment_seconds=4.0, n_fft=512, hop=128,
                  base_channels=48, depth=5, batch_size=8, steps_per_epoch=2000, epochs=200),
}

DEFAULTS: Dict[str, Any] = dict(
    clean_dir="data/datasets/clean",
    noise_dir="data/datasets/noise",
    sample_rate=16000,
    segment_seconds=2.0,
    snr_range=(-5.0, 20.0),
    n_fft=512,
    hop=128,
    base_channels=32,
    depth=4,
    use_gru=True,
    mask_type="sigmoid",
    batch_size=8,
    epochs=100,
    steps_per_epoch=1000,
    val_items=200,
    lr=3e-4,
    weight_decay=0.0,
    grad_clip=5.0,
    w_sisnr=1.0,
    w_stft=0.5,
    num_workers=0,
    seed=0,
    amp=True,
    out_dir="models/checkpoints",
    run_name="denoise_unet",
    log_every=25,
)


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path:
        import yaml

        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    return cfg


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--estimate-only", action="store_true",
                    help="time a handful of steps, print the ETA and exit without training")
    ap.add_argument("--estimate-json", default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is required for training:  pip install -r requirements-train.txt")
        return 1

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        # tensorboard is for logging only -- never a reason to refuse to train
        print("note: tensorboard not installed, scalar logging disabled "
              "(pip install tensorboard)")

        class SummaryWriter:  # type: ignore[no-redef]
            def __init__(self, *a, **k): pass
            def add_scalar(self, *a, **k): pass
            def close(self): pass

    from src.denoising.architectures import build_model
    from src.denoising.training.dataset import build_dataloaders
    from src.denoising.training.losses import CombinedLoss, si_snr

    cfg = load_config(args.config)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for key in ("clean_dir", "noise_dir", "epochs", "batch_size", "lr"):  # CLI wins over yaml
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if any(int(cfg[key]) <= 0 for key in ('epochs', 'steps_per_epoch', 'batch_size', 'val_items')):
        raise ValueError('epochs, steps_per_epoch, batch_size, and val_items must be positive')

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    # ---------------- model ------------------------------------------------ #
    model = build_model(cfg).to(device)
    params_m = model.n_params / 1e6
    print("=" * 74)
    print(" SpectralUNet -- %.2f M parameters | %d Hz | %.1f s crops | device=%s"
          % (params_m, cfg["sample_rate"], cfg["segment_seconds"], device))
    print("=" * 74)

    train_loader, val_loader = build_dataloaders(cfg)
    criterion = CombinedLoss(cfg["w_sisnr"], cfg["w_stft"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_si_sdr = 0, -1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_si_sdr = float(ckpt.get("best_val_si_sdr", ckpt.get("val_si_sdr", -1e9)))
        print(" resumed from %s at epoch %d (best %.2f dB)" % (args.resume, start_epoch, best_si_sdr))

    # ---------------- measured ETA ----------------------------------------- #
    print("\n timing the first steps to estimate the real cost...")
    from src.core.training_runtime import measured_estimate, report_remaining, write_estimate
    estimate = measured_estimate(model, optimiser, scaler,
        lambda: _time_steps(model, criterion, optimiser, scaler, train_loader, device, use_amp),
        lambda batches: _validate(model, batches, device, si_snr), val_loader, cfg,
        start_epoch, args.estimate_json)
    eta_hours = estimate['remaining_seconds'] / 3600
    if args.estimate_only:
        print("\n --estimate-only: stopping before training. Nothing was trained.")
        return 0
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(ROOT / "runs" / cfg["run_name"]))
    training_start = time.monotonic()
    if eta_hours > 24:
        print("\n ! this run would take longer than a day. Consider --preset tiny,")
        print("   fewer steps_per_epoch, or a GPU. Starting anyway in 5 s (ctrl-C to abort)...")
        time.sleep(5)

    # ---------------- train ------------------------------------------------ #
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    global_step = start_epoch * cfg["steps_per_epoch"]
    stop = False

    for epoch in range(start_epoch, cfg["epochs"]):
        train_loader.dataset.epoch = epoch
        model.train()
        epoch_start, running = time.time(), 0.0
        for step, (noisy, clean) in enumerate(train_loader):
            if step >= cfg["steps_per_epoch"]:
                break
            noisy, clean = noisy.to(device), clean.to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                est = model(noisy)
                loss, parts = criterion(est, clean)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimiser)
            scaler.update()

            running += float(loss.detach())
            global_step += 1
            if step % cfg["log_every"] == 0:
                writer.add_scalar("train/loss", float(loss.detach()), global_step)
                writer.add_scalar("train/si_snr_db", parts["sisnr_db"], global_step)
                print("  epoch %3d | step %5d/%d | loss %8.4f | SI-SNR %6.2f dB | %.2f s/step"
                      % (epoch + 1, step + 1, cfg["steps_per_epoch"], float(loss.detach()),
                         parts["sisnr_db"], (time.time() - epoch_start) / max(step + 1, 1)))
            if deadline and time.time() > deadline:
                print("\n --max-hours reached, wrapping up.")
                stop = True
                break

        val = _validate(model, val_loader, device, si_snr)
        scheduler.step(val["si_sdr"])
        writer.add_scalar("val/si_sdr_db", val["si_sdr"], epoch)
        writer.add_scalar("val/improvement_db", val["improvement"], epoch)
        print(" epoch %3d done in %s | train loss %.4f | val SI-SDR %.2f dB (input %.2f dB, +%.2f)"
              % (epoch + 1, _fmt_hours((time.time() - epoch_start) / 3600.0),
                 running / max(1, step + 1),
                 val["si_sdr"], val["input_si_sdr"], val["improvement"]))

        payload = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {**cfg, "arch": "SpectralUNet"},
            # A time-budget stop must not mark an unfinished epoch complete.
            # Resume continues from these weights and repeats the partial epoch.
            "epoch": epoch + 1 if step + 1 >= cfg['steps_per_epoch'] else epoch,
            "steps_in_epoch": step + 1,
            "val_si_sdr": val["si_sdr"],
            "best_val_si_sdr": max(best_si_sdr, val["si_sdr"]),
            "params_m": params_m,
        }
        torch.save(payload, out_dir / "denoise_unet_last.pt")
        if val["si_sdr"] > best_si_sdr:
            best_si_sdr = val["si_sdr"]
            torch.save(payload, out_dir / "denoise_unet_best.pt")
            print("   ^ new best -> %s" % (out_dir / "denoise_unet_best.pt"))
        if stop:
            break
        report_remaining(time.monotonic() - training_start, epoch + 1 - start_epoch,
                         cfg['epochs'] - epoch - 1)

    writer.close()
    if args.estimate_json:
        estimate.update(completed_epochs=payload['epoch'] if cfg['epochs'] > start_epoch else start_epoch,
                        stopped_early=stop, training_seconds=time.monotonic() - training_start)
        write_estimate(args.estimate_json, estimate)
    print("\n finished. best val SI-SDR %.2f dB" % best_si_sdr)
    print(" checkpoint directory: %s" % out_dir)
    return 0


# --------------------------------------------------------------------------- #
def _time_steps(model, criterion, optimiser, scaler, loader, device, use_amp, n_steps=8) -> Dict[str, float]:
    import torch

    model.train()
    def step(batch):
        noisy, clean = batch
        noisy, clean = noisy.to(device), clean.to(device)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss, _ = criterion(model(noisy), clean)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
    from src.core.training_runtime import time_batches
    return {"seconds_per_step": time_batches(step, loader, device, n_steps)}


def _validate(model, loader, device, si_snr_fn) -> Dict[str, float]:
    import torch

    model.eval()
    out_scores, in_scores = [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            est = model(noisy)
            n = min(est.shape[-1], clean.shape[-1])
            out_scores.append(si_snr_fn(est[..., :n], clean[..., :n]).mean().item())
            in_scores.append(si_snr_fn(noisy[..., :n], clean[..., :n]).mean().item())
    si = sum(out_scores) / max(len(out_scores), 1)
    base = sum(in_scores) / max(len(in_scores), 1)
    return {"si_sdr": si, "input_si_sdr": base, "improvement": si - base}


def _fmt_hours(hours: float) -> str:
    if hours < 1 / 60:
        return "%.0f s" % (hours * 3600)
    if hours < 1:
        return "%.0f min" % (hours * 60)
    if hours < 48:
        return "%.1f h" % hours
    return "%.1f days" % (hours / 24)


if __name__ == "__main__":
    raise SystemExit(main())
