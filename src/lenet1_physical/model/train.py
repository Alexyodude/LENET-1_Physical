"""Train LeNet-5 on MNIST. Run once, commit the resulting weights file."""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet1_physical.model.lenet import LeNet5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    parser.add_argument("--out", type=Path, default=Path("weights/lenet5.pt"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_ds)
        print(f"epoch {epoch + 1}: test_acc={acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
