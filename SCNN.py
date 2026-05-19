"""
Triplet-loss SCNN for battery fault diagnosis.

This script follows the manuscript setting:
- Binary similarity learning: normal vs abnormal
- Normal-OC and Normal-OD pairings
- Cycle-based split: cycles 1-80 for training, cycles 81-100 for testing
- SOC-wise evaluation: 20-30% to 70-80%
- Input image size: 64 x 64 x 3
- Embedding dimension: 128
- Optimizer: Adam
- Batch size: 32
- Epochs: 50
- Margin: alpha = 0.3 for GASF, alpha = 0.1 for MTF
- Random seeds: 42, 123, 456, 789, 1024
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


IMAGE_ROOT = Path("results/images")
OUTPUT_ROOT = Path("results/scnn")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
EMBEDDING_DIM = 128

SOC_LABELS = ["20_30", "30_40", "40_50", "50_60", "60_70", "70_80"]
MODES = ["charge", "discharge"]
PACK_TYPES = ["OC", "OD"]
TRANSFORMS = ["GASF", "MTF"]
DATA_TYPES = ["Voltage", "RLS"]

NORMAL_CELLS = [2, 3, 4, 5]
ABNORMAL_CELLS = [1, 6]

TRAIN_CYCLES = range(1, 81)
TEST_CYCLES = range(81, 101)

BATCH_SIZE = 32
EPOCHS = 50

MARGIN_MAP = {
    "GASF": 0.3,
    "MTF": 0.1,
}

SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_binary_dataset(pack_type, mode, transform, data_type, soc_label):
    """
    Load normal/abnormal binary dataset for one pack condition and SOC interval.

    Label definition:
    - 0: normal cells, Cells 2-5
    - 1: abnormal cells, Cells 1 and 6
    """
    folder = IMAGE_ROOT / pack_type / mode / transform / data_type / soc_label

    train_images, train_labels = [], []
    test_images, test_labels = [], []
    normal_test_pool = []

    for cycle in range(1, 101):
        for cell_idx in range(1, 7):
            img_path = folder / f"cycle{cycle:03d}_cell{cell_idx}.png"

            if not img_path.exists():
                continue

            image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

            if cell_idx in NORMAL_CELLS:
                label = 0
            elif cell_idx in ABNORMAL_CELLS:
                label = 1
            else:
                continue

            if cycle in TRAIN_CYCLES:
                train_images.append(image)
                train_labels.append(label)

            elif cycle in TEST_CYCLES:
                if label == 0:
                    normal_test_pool.append((image, label))
                else:
                    test_images.append(image)
                    test_labels.append(label)

    # Balanced normal test sampling: 40 normal samples from cycles 81-100
    if len(normal_test_pool) >= 40:
        selected_normal = random.sample(normal_test_pool, 40)
    else:
        selected_normal = normal_test_pool

    for image, label in selected_normal:
        test_images.append(image)
        test_labels.append(label)

    return (
        np.array(train_images, dtype=np.float32),
        np.array(train_labels, dtype=np.int64),
        np.array(test_images, dtype=np.float32),
        np.array(test_labels, dtype=np.int64),
    )


class EmbeddingNetwork(nn.Module):
    """
    SCNN embedding network.

    Backbone:
    Conv2D(32, 3x3) -> MaxPool
    Conv2D(64, 3x3) -> MaxPool
    Conv2D(128, 3x3) -> MaxPool
    Flatten(4608) -> Dense(256) -> Dropout(0.5) -> Dense(128)
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def triplet_loss(anchor, positive, negative, margin):
    d_ap = torch.sum((anchor - positive) ** 2, dim=1)
    d_an = torch.sum((anchor - negative) ** 2, dim=1)
    return torch.clamp(d_ap - d_an + margin, min=0.0).mean()


def generate_random_triplets(labels, max_triplets=1024):
    labels = np.asarray(labels)
    triplets = []

    for _ in range(max_triplets):
        anchor_label = np.random.choice(np.unique(labels))

        positive_pool = np.where(labels == anchor_label)[0]
        negative_pool = np.where(labels != anchor_label)[0]

        if len(positive_pool) < 2 or len(negative_pool) == 0:
            continue

        anchor_idx, positive_idx = np.random.choice(
            positive_pool,
            size=2,
            replace=False
        )
        negative_idx = np.random.choice(negative_pool)

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets


def generate_semi_hard_triplets(model, images, labels, margin, max_triplets=1024):
    """
    Semi-hard mining condition:
    d(a,p) < d(a,n) < d(a,p) + margin
    """
    model.eval()

    X = torch.tensor(images, dtype=torch.float32).to(DEVICE)
    y = np.asarray(labels)

    with torch.no_grad():
        embeddings = model(X).cpu().numpy()

    triplets = []

    for anchor_idx in range(len(y)):
        anchor_label = y[anchor_idx]

        positive_indices = np.where(y == anchor_label)[0]
        negative_indices = np.where(y != anchor_label)[0]

        positive_indices = positive_indices[positive_indices != anchor_idx]

        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue

        anchor_embedding = embeddings[anchor_idx]

        pos_distances = np.linalg.norm(
            embeddings[positive_indices] - anchor_embedding,
            axis=1
        ) ** 2

        neg_distances = np.linalg.norm(
            embeddings[negative_indices] - anchor_embedding,
            axis=1
        ) ** 2

        positive_idx = positive_indices[np.argmin(pos_distances)]
        d_ap = np.min(pos_distances)

        semi_hard_mask = np.logical_and(
            neg_distances > d_ap,
            neg_distances < d_ap + margin
        )

        if np.any(semi_hard_mask):
            candidate_negatives = negative_indices[semi_hard_mask]
            negative_idx = np.random.choice(candidate_negatives)
            triplets.append((anchor_idx, positive_idx, negative_idx))

        if len(triplets) >= max_triplets:
            break

    if len(triplets) == 0:
        triplets = generate_random_triplets(labels, max_triplets=max_triplets)

    return triplets


def classify_by_centroid(model, X_train, y_train, X_test):
    """
    Classify test samples based on distance to normal and abnormal centroids.
    """
    model.eval()

    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        train_embeddings = model(X_train_tensor).cpu().numpy()
        test_embeddings = model(X_test_tensor).cpu().numpy()

    normal_centroid = train_embeddings[y_train == 0].mean(axis=0)
    abnormal_centroid = train_embeddings[y_train == 1].mean(axis=0)

    dist_normal = np.linalg.norm(test_embeddings - normal_centroid, axis=1)
    dist_abnormal = np.linalg.norm(test_embeddings - abnormal_centroid, axis=1)

    return (dist_abnormal < dist_normal).astype(int)


def train_one_setting(pack_type, mode, transform, data_type, soc_label, seed):
    set_seed(seed)

    X_train, y_train, X_test, y_test = load_binary_dataset(
        pack_type=pack_type,
        mode=mode,
        transform=transform,
        data_type=data_type,
        soc_label=soc_label,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[SKIP] {pack_type} {mode} {transform}-{data_type} {soc_label}: no data")
        return None

    margin = MARGIN_MAP[transform]

    model = EmbeddingNetwork(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    train_losses = []

    for epoch in range(EPOCHS):
        model.train()

        triplets = generate_semi_hard_triplets(
            model=model,
            images=X_train,
            labels=y_train,
            margin=margin,
            max_triplets=1024,
        )

        random.shuffle(triplets)
        batch_losses = []

        for start in range(0, len(triplets), BATCH_SIZE):
            batch_triplets = triplets[start:start + BATCH_SIZE]

            anchor = torch.tensor(
                X_train[[t[0] for t in batch_triplets]],
                dtype=torch.float32
            ).to(DEVICE)

            positive = torch.tensor(
                X_train[[t[1] for t in batch_triplets]],
                dtype=torch.float32
            ).to(DEVICE)

            negative = torch.tensor(
                X_train[[t[2] for t in batch_triplets]],
                dtype=torch.float32
            ).to(DEVICE)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = triplet_loss(
                emb_anchor,
                emb_positive,
                emb_negative,
                margin=margin,
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_losses.append(float(np.mean(batch_losses)))

    y_pred = classify_by_centroid(model, X_train, y_train, X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(
        f"[DONE] {pack_type} {mode} {transform}-{data_type} {soc_label} "
        f"seed={seed} | Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}"
    )

    return {
        "pack_type": pack_type,
        "mode": mode,
        "transform": transform,
        "data_type": data_type,
        "soc": soc_label,
        "seed": seed,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "train_loss": train_losses,
    }


def run_all():
    results = []

    for pack_type in PACK_TYPES:
        for mode in MODES:
            for transform in TRANSFORMS:
                for data_type in DATA_TYPES:
                    for soc_label in SOC_LABELS:
                        for seed in SEEDS:
                            result = train_one_setting(
                                pack_type=pack_type,
                                mode=mode,
                                transform=transform,
                                data_type=data_type,
                                soc_label=soc_label,
                                seed=seed,
                            )

                            if result is not None:
                                results.append(result)

    save_results(results)


def save_results(results):
    df = pd.DataFrame([
        {
            "pack_type": r["pack_type"],
            "mode": r["mode"],
            "transform": r["transform"],
            "data_type": r["data_type"],
            "soc": r["soc"],
            "seed": r["seed"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        }
        for r in results
    ])

    summary_path = OUTPUT_ROOT / "scnn_results_all_seeds.csv"
    df.to_csv(summary_path, index=False)

    grouped = df.groupby(
        ["pack_type", "mode", "transform", "data_type", "soc"]
    )[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])

    grouped_path = OUTPUT_ROOT / "scnn_results_mean_std.csv"
    grouped.to_csv(grouped_path)

    print(f"Saved: {summary_path}")
    print(f"Saved: {grouped_path}")


if __name__ == "__main__":
    run_all()