"""Train the transformer on generated toy shapes; no biological data included."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import make_synthetic_shapes, random_observation_mask
from .model import LandmarkTransformer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default="fossil2skeleton-toy.pt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shapes = make_synthetic_shapes(512, seed=args.seed)
    mask = random_observation_mask(512, shapes.shape[1], seed=args.seed + 1)
    x = np.nan_to_num(np.where(mask[..., None], shapes, np.nan), nan=0.0)
    ds = TensorDataset(torch.tensor(x), torch.tensor(mask), torch.tensor(shapes))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkTransformer(n_landmarks=shapes.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = count = 0
        for partial, observed, target in loader:
            partial, observed, target = partial.to(device), observed.to(device), target.to(device)
            pred = model(partial, observed)
            missing = ~observed
            loss = ((pred - target).square().sum(dim=-1) * missing).sum() / missing.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
            count += 1
        print(f"epoch {epoch:02d} loss={loss_sum / count:.6f}")
    torch.save({"state_dict": model.cpu().state_dict(), "n_landmarks": shapes.shape[1]}, args.output)
    print(f"Saved toy-only model to {args.output}")


if __name__ == "__main__":
    main()
