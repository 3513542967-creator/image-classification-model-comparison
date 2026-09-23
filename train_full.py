"""Full-data experiment with a held-out validation split and one final test."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import get_experiment_datasets
from models import build_model, count_parameters, estimate_macs

MODELS = ["resnet18", "resnet34", "mobilenet_v3_small", "vit_tiny"]
DATASETS = ["cifar10", "tiny_imagenet"]
FIELDS = ["epoch", "learning_rate", "train_loss", "train_accuracy", "valid_loss",
          "valid_accuracy", "epoch_seconds", "elapsed_seconds"]


def device_for(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def one_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_total = correct = seen = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_total += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
    return loss_total / seen, correct / seen


def write_history(rows, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    dataset_train, dataset_valid, dataset_test, classes = get_experiment_datasets(
        args.dataset, args.data_dir, args.image_size, args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_options)
    valid_loader = DataLoader(dataset_valid, shuffle=False, **loader_options)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_options)
    model = build_model(args.model, len(classes), args.image_size).to(device)
    parameters = count_parameters(model)
    macs = estimate_macs(model, args.image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    result_dir = Path(args.output_dir) / args.dataset / args.model
    weight_dir = Path(args.checkpoint_dir) / args.dataset / args.model
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    history_path = result_dir / "history.csv"
    last_path = weight_dir / "last.pt"
    best_path = weight_dir / "best.pt"
    summary_path = result_dir / "summary.json"
    config = {"dataset": args.dataset, "model": args.model, "seed": args.seed,
              "image_size": args.image_size, "batch_size": args.batch_size,
              "max_epochs": args.epochs, "min_epochs": args.min_epochs,
              "patience": args.patience, "lr": args.lr}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["config"] == config:
            print(f"Already complete: {summary_path}", flush=True)
            return
        raise RuntimeError(f"Existing result has different settings: {summary_path}")

    history = []
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    first_epoch = 1
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["config"] != config:
            raise RuntimeError(f"Existing checkpoint has different settings: {last_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_acc, best_epoch, stale = state["best_acc"], state["best_epoch"], state["stale"]
        first_epoch = state["epoch"] + 1
        print(f"Resuming {args.dataset}/{args.model} at epoch {first_epoch}", flush=True)
    elif history_path.exists():
        raise RuntimeError(f"History exists without a resumable checkpoint: {history_path}")

    stop_reason = "max_epochs"
    print(f"{args.dataset}/{args.model}: train={len(dataset_train)}, valid={len(dataset_valid)}, "
          f"test={len(dataset_test)}, device={device}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = one_epoch(model, train_loader, optimizer, device)
        valid_loss, valid_accuracy = one_epoch(model, valid_loader, None, device)
        if device.type == "mps":
            torch.mps.synchronize()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        elapsed = duration + (history[-1]["elapsed_seconds"] if history else 0)
        history.append(dict(epoch=epoch, learning_rate=learning_rate,
                            train_loss=train_loss, train_accuracy=train_accuracy,
                            valid_loss=valid_loss, valid_accuracy=valid_accuracy,
                            epoch_seconds=duration, elapsed_seconds=elapsed))
        if valid_accuracy > best_acc + 1e-4:
            best_acc, best_epoch, stale = valid_accuracy, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        scheduler.step(valid_accuracy)
        write_history(history, history_path)
        torch.save(dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), history=history, best_acc=best_acc,
                        best_epoch=best_epoch, stale=stale, epoch=epoch), last_path)
        print(f"epoch={epoch:02d} train={train_accuracy:.4f} valid={valid_accuracy:.4f} "
              f"best={best_acc:.4f} time={duration:.1f}s lr={learning_rate:.5f}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"Validation plateau: {stale} epochs without improvement", flush=True)
            stop_reason = "validation_plateau"
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_accuracy = one_epoch(model, test_loader, None, device)
    summary = dict(config=config, train_images=len(dataset_train), valid_images=len(dataset_valid),
                   test_images=len(dataset_test), parameters=parameters, macs_per_image=macs,
                   flops_per_image=macs * 2, epochs_completed=len(history), best_epoch=best_epoch,
                   best_valid_accuracy=best_acc, test_accuracy=test_accuracy, test_loss=test_loss,
                   training_seconds=history[-1]["elapsed_seconds"], device=str(device),
                   checkpoint=str(best_path), history=str(history_path),
                   stop_reason=stop_reason)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"FINAL {args.dataset}/{args.model}: test_accuracy={test_accuracy:.4f} "
          f"best_epoch={best_epoch} epochs={len(history)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/full")
    parser.add_argument("--checkpoint-dir", default="checkpoints/full")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
