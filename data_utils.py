"""Dataset loading helpers kept in one place for beginner-friendly experiments."""
import os
import urllib.request
import zipfile
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root) / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        self.classes = sorted((self.root / "wnids.txt").read_text().splitlines())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        if split == "train":
            for name in self.classes:
                for path in (self.root / "train" / name / "images").glob("*.JPEG"):
                    self.samples.append((path, self.class_to_idx[name]))
        else:
            annotations = self.root / "val" / "val_annotations.txt"
            labels = {line.split("\t")[0]: self.class_to_idx[line.split("\t")[1]]
                      for line in annotations.read_text().splitlines()}
            for path in (self.root / "val" / "images").glob("*.JPEG"):
                if path.name in labels:
                    self.samples.append((path, labels[path.name]))
        if not self.samples:
            raise RuntimeError(f"No Tiny ImageNet {split} images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def ensure_tiny_imagenet(root):
    root = Path(root)
    target = root / "tiny-imagenet-200"
    if target.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "tiny-imagenet-200.zip"
    if not archive.exists():
        print("Downloading Tiny ImageNet (~237 MB)...")
        urllib.request.urlretrieve(TINY_URL, archive)
    print("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(root)
    archive.unlink(missing_ok=True)


def get_datasets(name, data_dir="data", image_size=224):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize,
    ])
    if name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        valid = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)
        class_names = train.classes
    elif name == "tiny_imagenet":
        ensure_tiny_imagenet(data_dir)
        train = TinyImageNet(data_dir, "train", train_transform)
        valid = TinyImageNet(data_dir, "val", eval_transform)
        class_names = train.classes
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, valid, class_names
