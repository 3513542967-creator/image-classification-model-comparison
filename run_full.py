"""Run all eight full-data experiments sequentially, resuming finished work."""
import subprocess
import sys
from pathlib import Path

DATASETS = ["cifar10", "tiny_imagenet"]
MODELS = ["mobilenet_v3_small", "resnet18", "resnet34", "vit_tiny"]


def main():
    for dataset in DATASETS:
        for model in MODELS:
            print(f"\nStarting {dataset}/{model}", flush=True)
            subprocess.run([sys.executable, "train_full.py", "--dataset", dataset,
                            "--model", model], check=True)
    subprocess.run([sys.executable, "analyze_full.py"], check=True)


if __name__ == "__main__":
    main()
