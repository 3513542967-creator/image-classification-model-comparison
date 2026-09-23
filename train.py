"""Train one image classifier and save weights, logs, and learning curves."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import get_datasets
from models import build_model, count_parameters, estimate_macs

MODELS = ["resnet18", "resnet34", "mobilenet_v3_small", "vit_tiny"]
DATASETS = ["cifar10", "tiny_imagenet"]


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, optimizer, device, training):
    model.train(training)
    loss_sum = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count


def save_curves(rows, path, title):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(epochs, [r["valid_loss"] for r in rows], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(epochs, [100*r["train_accuracy"] for r in rows], label="train")
    axes[1].plot(epochs, [100*r["valid_accuracy"] for r in rows], label="validation")
    axes[1].set(title="Accuracy (%)", xlabel="Epoch"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")
    train_data, valid_data, classes = get_datasets(args.dataset, args.data_dir)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)
    model = build_model(args.model, len(classes)).to(device)
    params = count_parameters(model)
    macs = estimate_macs(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log_dir = Path(args.output_dir) / "logs"; figure_dir = Path(args.output_dir) / "figures"
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []; best_acc = 0.0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        valid_loss, valid_acc = run_epoch(model, valid_loader, loss_fn, optimizer, device, False)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "valid_loss": valid_loss, "valid_accuracy": valid_acc,
               "epoch_seconds": time.time() - epoch_start}
        rows.append(row)
        print(f"Epoch {epoch}/{args.epochs}: val_acc={valid_acc:.4f}, val_loss={valid_loss:.4f}")
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save({"model": model.state_dict(), "model_name": args.model,
                        "dataset": args.dataset, "classes": classes,
                        "image_size": 224, "epoch": epoch, "valid_accuracy": valid_acc,
                        "seed": args.seed}, checkpoint_dir / f"{args.model}.pt")
    elapsed = time.time() - started
    log_path = log_dir / f"{args.dataset}_{args.model}.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    save_curves(rows, figure_dir / f"{args.dataset}_{args.model}.png", f"{args.model} on {args.dataset}")
    result = {"dataset": args.dataset, "model": args.model, "classes": len(classes),
              "epochs": args.epochs, "best_valid_accuracy": best_acc, "parameters": params,
              "macs_per_image": macs, "flops_per_image": macs * 2,
              "training_seconds": elapsed, "device": str(device), "seed": args.seed,
              "checkpoint": str(checkpoint_dir / f"{args.model}.pt"), "log": str(log_path)}
    json_path = log_dir / f"{args.dataset}_{args.model}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__": main()
