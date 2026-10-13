#!/usr/bin/env python3
"""Pretrain reproducible ComplEx structure initializers for a WN18-family dataset.

This is intentionally independent of text and images: its outputs are used only
as the structural initialization consumed by ReLook's progressive backbone.
Both original and reciprocal relations are trained, matching ReLook's 2R
relation table.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "data/processed/WN18RR++",
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=project_root / "img_data"
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Feature filename prefix; defaults to the dataset directory name.",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41504)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def read_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return int(stream.readline().strip())


def read_triples(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as stream:
        expected = int(stream.readline().strip())
        triples = [[int(value) for value in line.split()] for line in stream if line.strip()]
    if len(triples) != expected:
        raise ValueError(f"{path}: header says {expected}, found {len(triples)}")
    # Source format is head, tail, relation. Training format is head, relation, tail.
    array = torch.tensor(triples, dtype=torch.long)
    return array[:, [0, 2, 1]].contiguous()


class ComplexInitializer(torch.nn.Module):
    def __init__(self, n_entities: int, n_relations_2x: int, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("ComplEx output dimension must be even")
        self.half_dim = dim // 2
        self.entity_re = torch.nn.Embedding(n_entities, self.half_dim)
        self.entity_im = torch.nn.Embedding(n_entities, self.half_dim)
        self.relation_re = torch.nn.Embedding(n_relations_2x, self.half_dim)
        self.relation_im = torch.nn.Embedding(n_relations_2x, self.half_dim)
        bound = math.sqrt(6.0 / self.half_dim)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -bound, bound)

    @staticmethod
    def score_parts(
        h_re: torch.Tensor,
        h_im: torch.Tensor,
        r_re: torch.Tensor,
        r_im: torch.Tensor,
        t_re: torch.Tensor,
        t_im: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (h_re * r_re - h_im * r_im) * t_re
            + (h_re * r_im + h_im * r_re) * t_im
        ).sum(dim=-1)

    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor):
        return self.score_parts(
            self.entity_re(heads),
            self.entity_im(heads),
            self.relation_re(relations),
            self.relation_im(relations),
            self.entity_re(tails),
            self.entity_im(tails),
        )

    def score_negative_tails(
        self, heads: torch.Tensor, relations: torch.Tensor, negative_tails: torch.Tensor
    ) -> torch.Tensor:
        return self.score_parts(
            self.entity_re(heads).unsqueeze(1),
            self.entity_im(heads).unsqueeze(1),
            self.relation_re(relations).unsqueeze(1),
            self.relation_im(relations).unsqueeze(1),
            self.entity_re(negative_tails),
            self.entity_im(negative_tails),
        )

    def regularizer(
        self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor
    ) -> torch.Tensor:
        factors = (
            self.entity_re(heads), self.entity_im(heads),
            self.relation_re(relations), self.relation_im(relations),
            self.entity_re(tails), self.entity_im(tails),
        )
        return sum(value.abs().pow(3).mean() for value in factors)

    def export(self) -> Tuple[np.ndarray, np.ndarray]:
        entity = torch.cat([self.entity_re.weight, self.entity_im.weight], dim=1)
        relation = torch.cat([self.relation_re.weight, self.relation_im.weight], dim=1)
        entity = F.normalize(entity, p=2, dim=1)
        return (
            entity.detach().cpu().float().numpy(),
            relation.detach().cpu().float().numpy(),
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    feature_dir = args.feature_dir.expanduser().resolve()
    n_entities = read_count(dataset_dir / "entity2id.txt")
    n_relations = read_count(dataset_dir / "relation2id.txt")
    triples = read_triples(dataset_dir / "train2id.txt")
    inverse = torch.stack(
        [triples[:, 2], triples[:, 1] + n_relations, triples[:, 0]], dim=1
    )
    triples = torch.cat([triples, inverse], dim=0)

    device = torch.device(args.device)
    model = ComplexInitializer(n_entities, n_relations * 2, args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    losses = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(triples), generator=generator)
        epoch_loss = 0.0
        examples = 0
        model.train()
        for start in range(0, len(triples), args.batch_size):
            batch = triples[permutation[start : start + args.batch_size]].to(device)
            heads, relations, tails = batch.unbind(dim=1)
            negative_tails = torch.randint(
                n_entities,
                (len(batch), args.negatives),
                device=device,
            )
            # Avoid the known target itself; other true answers are rare and are
            # not available to this lightweight initializer's sampled objective.
            target_collision = negative_tails.eq(tails.unsqueeze(1))
            while target_collision.any():
                negative_tails[target_collision] = torch.randint(
                    n_entities,
                    (int(target_collision.sum()),),
                    device=device,
                )
                target_collision = negative_tails.eq(tails.unsqueeze(1))

            positive_score = model.score(heads, relations, tails)
            negative_score = model.score_negative_tails(
                heads, relations, negative_tails
            )
            negative_weight = torch.softmax(
                negative_score.detach() * args.adversarial_temperature, dim=1
            )
            loss = F.softplus(-positive_score).mean()
            loss = loss + (negative_weight * F.softplus(negative_score)).sum(1).mean()
            if args.regularization:
                loss = loss + args.regularization * model.regularizer(
                    heads, relations, tails
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            examples += len(batch)

        mean_loss = epoch_loss / examples
        losses.append(mean_loss)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"epoch={epoch:03d} loss={mean_loss:.6f}", flush=True)

    entity_features, relation_features = model.export()
    feature_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or dataset_dir.name
    with (feature_dir / f"{dataset_name}_entity_embed.pkl").open("wb") as stream:
        pickle.dump(entity_features, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (feature_dir / f"{dataset_name}_relation_embed.pkl").open("wb") as stream:
        pickle.dump(relation_features, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "dataset": dataset_name,
        "method": "ComplEx sampled-logistic structural pretraining with reciprocal relations",
        "n_entities": n_entities,
        "n_relations_with_reciprocals": n_relations * 2,
        "dim": args.dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "negatives": args.negatives,
        "learning_rate": args.lr,
        "regularization": args.regularization,
        "seed": args.seed,
        "final_loss": losses[-1],
        "losses": losses,
    }
    with (feature_dir / f"{dataset_name}_structure_pretrain.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
