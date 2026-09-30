"""Train the local Conv-TasNet separator.

    python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml

As with the denoiser: the app never starts this. Run it yourself, and run it
with ``--estimate-only`` first -- it times real steps on your machine and prints
the true ETA instead of a guess.

    --preset tiny|base|paper   size shortcut
    --n-src N                  2 or 3 speakers (baked into the checkpoint)
    --estimate-only            measure, print ETA, exit
    --max-hours H              stop cleanly after H hours
    --resume PATH              continue a run

Outputs
-------
    models/checkpoints/separation_convtasnet_last.pt
    models/checkpoints/separation_convtasnet_best.pt   <- the app loads this
    runs/separation_convtasnet/...                       tensorboard
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
    # ~0.9 M params, 8 kHz, 2 s crops -- the only one that is sane on a CPU
    "tiny": dict(sample_rate=8000, segment_seconds=2.0, enc_channels=128, enc_kernel=16,
                 bottleneck=64, hidden=128, n_blocks=5, n_repeats=2,
                 batch_size=8, steps_per_epoch=400, epochs=40),
    # ~3.5 M params -- the project default
    "base": dict(sample_rate=8000, segment_seconds=4.0, enc_channels=256, enc_kernel=16,
                 bottleneck=128, hidden=256, n_blocks=7, n_repeats=2,
                 batch_size=4, steps_per_epoch=1000, epochs=100),
    # the published Conv-TasNet configuration (N=512, X=8, R=3) -- GPU only
    "paper": dict(sample_rate=8000, segment_seconds=4.0, enc_channels=512, enc_kernel=16,
                  bottleneck=128, hidden=512, n_blocks=8, n_repeats=3,
                  batch_size=4, steps_per_epoch=2000, epochs=200),
}

DEFAULTS: Dict[str, Any] = dict(
    clean_dir="data/datasets/speakers",
    noise_dir=None,
    n_src=2,
    sample_rate=8000,
    segment_seconds=4.0,
    enc_channels=256,
    enc_kernel=16,
    bottleneck=128,
    hidden=256,
    kernel=3,
    n_blocks=7,
    n_repeats=2,
    norm="gLN",
    causal=False,
    mask_act="relu",
    batch_size=4,
    epochs=100,
    steps_per_epoch=1000,
    val_items=150,
    lr=1e-3,
    weight_decay=0.0,
    grad_clip=5.0,
    mixture_consistency=0.1,
    num_workers=0,
    seed=0,
    amp=True,
    out_dir="models/checkpoints",
    run_name="separation_convtasnet",
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
    ap.add_argument("--n-src", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--estimate-only", action="store_true")
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

    from src.separation.architectures import build_model
    from src.separation.training.dataset import build_dataloaders
    from src.separation.training.losses import MixtureConsistencyLoss, PITLossWrapper

    cfg = load_config(args.config)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for key in ("clean_dir", "noise_dir", "n_src", "epochs", "batch_size", "lr"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if any(int(cfg[key]) <= 0 for key in ('epochs', 'steps_per_epoch', 'batch_size', 'val_items')):
        raise ValueError('epochs, steps_per_epoch, batch_size, and val_items must be positive')

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    model = build_model(cfg).to(device)
    params_m = model.n_params / 1e6
    print("=" * 74)
    print(" Conv-TasNet -- %.2f M params | %d sources | %d Hz | %.1f s crops | device=%s"
          % (params_m, cfg["n_src"], cfg["sample_rate"], cfg["segment_seconds"], device))
    print("=" * 74)

    train_loader, val_loader = build_dataloaders(cfg)
    criterion = PITLossWrapper(cfg["n_src"]).to(device)
    consistency = MixtureConsistencyLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best = 0, -1e9
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
        best = float(ckpt.get("best_val_si_sdr", ckpt.get("val_si_sdr", -1e9)))
        print(" resumed from %s at epoch %d (best %.2f dB)" % (args.resume, start_epoch, best))

    # ---------------- measured ETA ----------------------------------------- #
    print("\n timing the first steps to estimate the real cost...")
    from src.core.training_runtime import measured_estimate, report_remaining, write_estimate
    estimate = measured_estimate(model, optimiser, scaler,
        lambda: _time_steps(model, criterion, consistency, optimiser, scaler, train_loader, device, use_amp, cfg),
        lambda batches: _validate(model, criterion, batches, device), val_loader, cfg,
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
        print("\n ! longer than a day. Consider --preset tiny, fewer steps, or a GPU.")
        print("   starting in 5 s (ctrl-C to abort)...")
        time.sleep(5)

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    global_step = start_epoch * cfg["steps_per_epoch"]
    stop = False

    for epoch in range(start_epoch, cfg["epochs"]):
        train_loader.dataset.epoch = epoch
        model.train()
        epoch_start, running = time.time(), 0.0
        for step, (mixture, sources) in enumerate(train_loader):
            if step >= cfg["steps_per_epoch"]:
                break
            mixture, sources = mixture.to(device), sources.to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                est = model(mixture)
                loss, parts = criterion(est, sources)
                if cfg["mixture_consistency"]:
                    loss = loss + cfg["mixture_consistency"] * consistency(est, mixture)
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
                writer.add_scalar("train/si_snr_db", parts["si_snr_db"], global_step)
                print("  epoch %3d | step %5d/%d | loss %8.4f | SI-SNR %6.2f dB | %.2f s/step"
                      % (epoch + 1, step + 1, cfg["steps_per_epoch"], float(loss.detach()),
                         parts["si_snr_db"], (time.time() - epoch_start) / max(step + 1, 1)))
            if deadline and time.time() > deadline:
                print("\n --max-hours reached, wrapping up.")
                stop = True
                break

        val = _validate(model, criterion, val_loader, device)
        scheduler.step(val["si_sdr"])
        writer.add_scalar("val/si_sdr_db", val["si_sdr"], epoch)
        writer.add_scalar("val/si_sdri_db", val["si_sdri"], epoch)
        print(" epoch %3d done in %s | train loss %.4f | val SI-SDR %.2f dB | SI-SDRi %+.2f dB"
              % (epoch + 1, _fmt_hours((time.time() - epoch_start) / 3600.0),
                 running / max(1, step + 1), val["si_sdr"], val["si_sdri"]))

        payload = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {**cfg, "arch": "ConvTasNet"},
            # Resume from partial weights without skipping the unfinished epoch.
            "epoch": epoch + 1 if step + 1 >= cfg['steps_per_epoch'] else epoch,
            "steps_in_epoch": step + 1,
            "val_si_sdr": val["si_sdr"],
            "best_val_si_sdr": max(best, val["si_sdr"]),
            "val_si_sdri": val["si_sdri"],
            "params_m": params_m,
        }
        torch.save(payload, out_dir / "separation_convtasnet_last.pt")
        if val["si_sdr"] > best:
            best = val["si_sdr"]
            torch.save(payload, out_dir / "separation_convtasnet_best.pt")
            print("   ^ new best -> %s" % (out_dir / "separation_convtasnet_best.pt"))
        if stop:
            break
        report_remaining(time.monotonic() - training_start, epoch + 1 - start_epoch,
                         cfg['epochs'] - epoch - 1)

    writer.close()
    if args.estimate_json:
        estimate.update(completed_epochs=payload['epoch'] if cfg['epochs'] > start_epoch else start_epoch,
                        stopped_early=stop, training_seconds=time.monotonic() - training_start)
        write_estimate(args.estimate_json, estimate)
    print("\n finished. best val SI-SDR %.2f dB" % best)
    print(" checkpoint directory: %s" % out_dir)
    return 0


# --------------------------------------------------------------------------- #
def _time_steps(model, criterion, consistency, optimiser, scaler, loader, device, use_amp, cfg, n_steps=8) -> float:
    import torch

    model.train()
    def step(batch):
        mixture, sources = batch
        mixture, sources = mixture.to(device), sources.to(device)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            est = model(mixture)
            loss, _ = criterion(est, sources)
            if cfg["mixture_consistency"]:
                loss = loss + cfg["mixture_consistency"] * consistency(est, mixture)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
    from src.core.training_runtime import time_batches
    return time_batches(step, loader, device, n_steps)


def _validate(model, criterion, loader, device) -> Dict[str, float]:
    import torch

    from src.separation.training.losses import si_snr

    model.eval()
    outs, bases = [], []
    with torch.no_grad():
        for mixture, sources in loader:
            mixture, sources = mixture.to(device), sources.to(device)
            est = model(mixture)
            loss, parts = criterion(est, sources)
            outs.append(parts["si_snr_db"])
            # baseline: the unseparated mixture scored against each reference
            n = min(mixture.shape[-1], sources.shape[-1])
            repeated = mixture[..., :n].unsqueeze(1).expand(-1, sources.shape[1], -1)
            bases.append(float(si_snr(repeated, sources[..., :n]).mean()))
    si = sum(outs) / max(len(outs), 1)
    base = sum(bases) / max(len(bases), 1)
    return {"si_sdr": si, "input_si_sdr": base, "si_sdri": si - base}


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
