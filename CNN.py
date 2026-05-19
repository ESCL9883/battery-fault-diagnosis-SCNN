"""
CNN classifier for battery fault diagnosis using GASF/MTF images.

This script follows the manuscript setting:
- 18-class classification: A-R
- Cycle-based split: cycles 1-80 for training, cycles 81-100 for testing
- Input image size: 64 x 64 x 3
- Optimizer: Adam
- Batch size: 32
- Epochs: 30

Expected directory structure:
results/images/
  OC/
    charge/GASF/RLS/20_30/cycle001_cell1.png
    discharge/GASF/RLS/20_30/cycle001_cell1.png
  OD/
    charge/GASF/RLS/20_30/cycle001_cell1.png
    discharge/GASF/RLS/20_30/cycle001_cell1.png
"""

from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_ROOT = Path("results/images")
IMAGE_SIZE = (64, 64)

PACK_TYPES = ["OC", "OD"]
MODES = ["charge", "discharge"]
TRANSFORMS = ["GASF", "MTF"]
DATA_TYPES = ["Voltage", "RLS"]

SOC_LABELS = ["20_30", "30_40", "40_50", "50_60", "60_70", "70_80"]

CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F",      # Normal
    "G", "H", "I", "J", "K", "L",      # Overcharge abnormal
    "M", "N", "O", "P", "Q", "R",      # Overdischarge abnormal
]

NORMAL_CELLS = [2, 3, 4, 5]
ABNORMAL_CELLS = [1, 6]

TRAIN_CYCLES = range(1, 81)
TEST_CYCLES = range(81, 101)

BATCH_SIZE = 32
EPOCHS = 30


def get_class_label(pack_type, soc_idx, cell_idx):
    if cell_idx in NORMAL_CELLS:
        return CLASS_LABELS[soc_idx]

    if cell_idx in ABNORMAL_CELLS and pack_type == "OC":
        return CLASS_LABELS[6 + soc_idx]

    if cell_idx in ABNORMAL_CELLS and pack_type == "OD":
        return CLASS_LABELS[12 + soc_idx]

    raise ValueError("Invalid cell or pack type.")


def load_dataset(mode="charge", transform="GASF", data_type="RLS"):
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    normal_test_pool = {soc: [] for soc in SOC_LABELS}

    for pack_type in PACK_TYPES:
        for soc_idx, soc in enumerate(SOC_LABELS):
            folder = IMAGE_ROOT / pack_type / mode / transform / data_type / soc

            for cycle in range(1, 101):
                for cell_idx in range(1, 7):
                    img_path = folder / f"cycle{cycle:03d}_cell{cell_idx}.png"

                    if not img_path.exists():
                        continue

                    image = load_img(img_path, target_size=IMAGE_SIZE)
                    image = img_to_array(image) / 255.0

                    label = get_class_label(pack_type, soc_idx, cell_idx)

                    if cycle in TRAIN_CYCLES:
                        train_images.append(image)
                        train_labels.append(label)

                    elif cycle in TEST_CYCLES:
                        if cell_idx in NORMAL_CELLS:
                            normal_test_pool[soc].append((image, label))
                        else:
                            test_images.append(image)
                            test_labels.append(label)

    # Match manuscript: normal test samples are randomly drawn from cycles 81-100
    # to ensure balanced evaluation with abnormal test samples.
    for soc in SOC_LABELS:
        pool = normal_test_pool[soc]
        if len(pool) >= 40:
            selected = random.sample(pool, 40)
        else:
            selected = pool

        for image, label in selected:
            test_images.append(image)
            test_labels.append(label)

    return (
        np.array(train_images, dtype=np.float32),
        np.array(train_labels),
        np.array(test_images, dtype=np.float32),
        np.array(test_labels),
    )


def build_cnn_classifier(input_shape=(64, 64, 3), num_classes=18):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01)
        ),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def train_cnn(mode="charge", transform="GASF", data_type="RLS"):
    X_train, y_train, X_test, y_test = load_dataset(
        mode=mode,
        transform=transform,
        data_type=data_type,
    )

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)
    y_test_int = label_encoder.transform(y_test)

    y_train_onehot = tf.keras.utils.to_categorical(y_train_int, num_classes=18)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_int, num_classes=18)

    model = build_cnn_classifier(input_shape=(64, 64, 3), num_classes=18)

    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(
        y_test_int,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    print(confusion_matrix(y_test_int, y_pred))

    return model, history


if __name__ == "__main__":
    train_cnn(
        mode="charge",
        transform="GASF",
        data_type="RLS",
    )