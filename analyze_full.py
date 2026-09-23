"""Summarize completed full-data runs and plot convergence and efficiency."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("results/full")


def main():
    summaries = []
    histories = {}
    for path in sorted(ROOT.glob("*/*/summary.json")):
        result = json.loads(path.read_text())
        history = pd.read_csv(path.parent / "history.csv")
        threshold = result["best_valid_accuracy"] * 0.95
        reached = history.loc[history.valid_accuracy >= threshold].iloc[0]
        stable = history.iloc[next(
            index for index in range(len(history))
            if (history.valid_accuracy.iloc[index:] >= threshold).all())]
        result["epoch_to_95pct_best"] = int(reached.epoch)
        result["minutes_to_95pct_best"] = float(reached.elapsed_seconds / 60)
        result["stable_epoch_at_95pct_best"] = int(stable.epoch)
        result["minutes_to_stable_95pct_best"] = float(stable.elapsed_seconds / 60)
        result["accuracy_per_million_params"] = result["test_accuracy"] / (result["parameters"] / 1e6)
        result["accuracy_per_gflop"] = result["test_accuracy"] / (result["flops_per_image"] / 1e9)
        result["dataset"] = result["config"]["dataset"]
        result["model"] = result["config"]["model"]
        chance_accuracy = 0.1 if result["dataset"] == "cifar10" else 0.005
        result["above_chance_accuracy_per_million_params"] = (
            result["test_accuracy"] - chance_accuracy) / (result["parameters"] / 1e6)
        summaries.append(result)
        histories[(result["dataset"], result["model"])] = history
    if not summaries:
        raise RuntimeError("No full experiment summaries found")
    table = pd.DataFrame(summaries)
    columns = ["dataset", "model", "train_images", "valid_images", "test_images",
               "test_accuracy", "best_valid_accuracy", "parameters", "flops_per_image",
               "epochs_completed", "best_epoch", "training_seconds", "epoch_to_95pct_best",
               "minutes_to_95pct_best", "stable_epoch_at_95pct_best",
               "minutes_to_stable_95pct_best", "above_chance_accuracy_per_million_params",
               "accuracy_per_gflop", "stop_reason", "device"]
    table[columns].to_csv(ROOT / "comparison.csv", index=False)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    for dataset, subset in table.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for _, item in subset.iterrows():
            history = histories[(dataset, item.model)]
            axes[0].plot(history.epoch, history.valid_accuracy * 100, label=item.model)
            axes[1].plot(history.elapsed_seconds / 60, history.valid_accuracy * 100, label=item.model)
        axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Convergence by epoch")
        axes[1].set(xlabel="Training time (minutes)", ylabel="Validation accuracy (%)",
                    title="Convergence by time")
        for axis in axes:
            axis.legend(fontsize=8); axis.grid(alpha=.25)
        fig.suptitle(dataset); fig.tight_layout()
        fig.savefig(figures / f"convergence_{dataset}.png", dpi=160); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].bar(subset.model, subset.test_accuracy * 100)
        axes[0].set(ylabel="Test accuracy (%)", title="Final held-out performance")
        axes[1].scatter(subset.parameters / 1e6, subset.test_accuracy * 100)
        axes[1].set(xlabel="Parameters (millions)", ylabel="Test accuracy (%)", title="Accuracy vs parameters")
        axes[2].scatter(subset.flops_per_image / 1e9, subset.test_accuracy * 100)
        axes[2].set(xlabel="FLOPs / image (billions)", ylabel="Test accuracy (%)", title="Accuracy vs compute")
        for _, item in subset.iterrows():
            axes[1].annotate(item.model, (item.parameters / 1e6, item.test_accuracy * 100), fontsize=8)
            axes[2].annotate(item.model, (item.flops_per_image / 1e9, item.test_accuracy * 100), fontsize=8)
        for axis in axes:
            axis.grid(alpha=.25)
        axes[0].tick_params(axis="x", rotation=25)
        fig.suptitle(dataset); fig.tight_layout()
        fig.savefig(figures / f"efficiency_{dataset}.png", dpi=160); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].bar(subset.model, subset.epochs_completed)
        axes[0].set(ylabel="Epochs", title="Training epochs")
        axes[1].bar(subset.model, subset.training_seconds / 3600)
        axes[1].set(ylabel="Hours", title="Total training time")
        axes[2].bar(subset.model, subset.above_chance_accuracy_per_million_params * 100)
        axes[2].set(ylabel="Percentage points / million parameters",
                    title="Above-chance accuracy per parameter")
        for axis in axes:
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=.25)
        fig.suptitle(dataset); fig.tight_layout()
        fig.savefig(figures / f"training_{dataset}.png", dpi=160); plt.close(fig)

        for _, item in subset.iterrows():
            history = histories[(dataset, item.model)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(history.epoch, history.train_accuracy * 100, label="train")
            axes[0].plot(history.epoch, history.valid_accuracy * 100, label="validation")
            axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
            axes[1].plot(history.epoch, history.train_loss, label="train")
            axes[1].plot(history.epoch, history.valid_loss, label="validation")
            axes[1].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
            for axis in axes:
                axis.legend(); axis.grid(alpha=.25)
            fig.suptitle(f"{dataset} / {item.model}")
            fig.tight_layout()
            fig.savefig(figures / f"learning_{dataset}_{item.model}.png", dpi=160)
            plt.close(fig)
    print(f"Saved full results to {ROOT / 'comparison.csv'}")


if __name__ == "__main__":
    main()
