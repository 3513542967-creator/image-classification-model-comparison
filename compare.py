"""Run the same training recipe for several models and datasets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train import MODELS, DATASETS, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        for model in args.models:
            print(f"\n{'='*68}\nTraining {model} on {dataset}\n{'='*68}")
            results.append(train(args=argparse.Namespace(**vars(args), dataset=dataset, model=model, lr=0.001)))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(output / "comparison.csv", index=False)
    (output / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset, group in table.groupby("dataset"):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].bar(group.model, group.best_valid_accuracy * 100)
        axes[0].set_title("Best validation accuracy (%)")
        axes[1].bar(group.model, group.parameters / 1e6)
        axes[1].set_title("Parameters (millions)")
        axes[2].bar(group.model, group.flops_per_image / 1e9)
        axes[2].set_title("FLOPs / image (billions)")
        axes[3].bar(group.model, group.training_seconds / 60)
        axes[3].set_title("Training time (minutes)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Model comparison on {dataset}"); fig.tight_layout()
        fig.savefig(figure_dir / f"comparison_{dataset}.png", dpi=160); plt.close(fig)
    print(f"\nSaved summary: {output / 'comparison.csv'}")

if __name__ == "__main__": main()
