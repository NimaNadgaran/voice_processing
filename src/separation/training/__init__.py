"""Training code for the local separator.

NOTHING here runs automatically. The web app and the CLI never import this
package; it exists so you can train
``models/checkpoints/separation_convtasnet_best.pt`` yourself:

    pip install -r requirements-train.txt
    python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml --estimate-only
    python -m src.separation.training.train_convtasnet --config src/separation/training/config.yaml

See ``src/separation/README.md`` -> "Training the local model" for the recipe,
the dataset links and the time estimates.
"""
