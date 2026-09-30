"""Training code for the local denoiser.

NOTHING here runs automatically. The web app and the CLI never import this
package; it exists so you can train ``models/checkpoints/denoise_unet_best.pt``
yourself when you want to:

    pip install -r requirements-train.txt
    python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml --estimate-only
    python -m src.denoising.training.train_unet --config src/denoising/training/config.yaml

See ``src/denoising/README.md`` -> "Training the local model" for the recipe,
the dataset links and the time estimates.
"""
