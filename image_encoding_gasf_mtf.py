"""
GASF/MTF image encoding from PAA-preprocessed battery time-series.

Pipeline:
1. Raw voltage and RLS-derived parameter time-series
2. PAA preprocessing to fixed length N = 50
3. GASF/MTF image encoding

This script assumes that PAA-preprocessed CSV files are already saved.
Required columns:
- Voltage
- Ri
- Rdiff
- Cdiff

For RLS parameter RGB images:
- Ri    -> Red channel
- Rdiff -> Green channel
- Cdiff -> Blue channel

No additional weighting is applied.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField


def minmax_to_uint8(image):
    image = np.asarray(image, dtype=float)

    image_min = np.min(image)
    image_max = np.max(image)

    if image_max - image_min < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)

    image_norm = (image - image_min) / (image_max - image_min)
    return (image_norm * 255).astype(np.uint8)


def create_transformer(method="gasf", image_size=50):
    method = method.lower()

    if method == "gasf":
        return GramianAngularField(image_size=image_size, method="summation")

    if method == "mtf":
        return MarkovTransitionField(image_size=image_size)

    raise ValueError("method must be either 'gasf' or 'mtf'.")


def encode_paa_series(series, method="gasf", image_size=50):
    """
    Convert one PAA-preprocessed time-series into a GASF or MTF image.
    """
    series = np.asarray(series, dtype=float)

    if len(series) != image_size:
        raise ValueError(
            f"PAA sequence length must be {image_size}, but got {len(series)}."
        )

    transformer = create_transformer(method=method, image_size=image_size)
    image = transformer.fit_transform(series.reshape(1, -1))[0]

    return minmax_to_uint8(image)


def encode_voltage_image(voltage_series, method="gasf", image_size=50):
    """
    Convert PAA-preprocessed voltage sequence into RGB image.
    """
    voltage_img = encode_paa_series(
        voltage_series,
        method=method,
        image_size=image_size,
    )

    return np.stack([voltage_img, voltage_img, voltage_img], axis=-1)


def encode_rls_rgb_image(ri_series, rdiff_series, cdiff_series,
                         method="gasf", image_size=50):
    """
    Convert PAA-preprocessed RLS parameters into RGB image.

    Ri    -> Red
    Rdiff -> Green
    Cdiff -> Blue
    """
    ri_img = encode_paa_series(
        ri_series,
        method=method,
        image_size=image_size,
    )

    rdiff_img = encode_paa_series(
        rdiff_series,
        method=method,
        image_size=image_size,
    )

    cdiff_img = encode_paa_series(
        cdiff_series,
        method=method,
        image_size=image_size,
    )

    return np.stack([ri_img, rdiff_img, cdiff_img], axis=-1)


def save_image(image_array, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(output_path)


def process_paa_condition(condition, paa_root, image_root,
                          methods=("gasf", "mtf"), image_size=50):
    """
    Load PAA-preprocessed CSV files and convert them into GASF/MTF images.

    Expected input:
    results/paa/{condition}/*.csv

    Output:
    results/images/{condition}/{method}/voltage/
    results/images/{condition}/{method}/rls_rgb/
    """
    input_dir = Path(paa_root) / condition
    output_dir = Path(image_root) / condition

    if not input_dir.exists():
        print(f"[SKIP] PAA directory not found: {input_dir}")
        return

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"[SKIP] No PAA CSV files found in: {input_dir}")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required_columns = ["Voltage", "Ri", "Rdiff", "Cdiff"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"[SKIP] {csv_file.name}: missing columns {missing_columns}")
            continue

        voltage = df["Voltage"].values
        ri = df["Ri"].values
        rdiff = df["Rdiff"].values
        cdiff = df["Cdiff"].values

        file_stem = csv_file.stem

        for method in methods:
            voltage_image = encode_voltage_image(
                voltage,
                method=method,
                image_size=image_size,
            )

            voltage_output = (
                output_dir
                / method
                / "voltage"
                / f"{file_stem}_voltage_{method}.png"
            )

            save_image(voltage_image, voltage_output)

            rls_rgb_image = encode_rls_rgb_image(
                ri,
                rdiff,
                cdiff,
                method=method,
                image_size=image_size,
            )

            rls_output = (
                output_dir
                / method
                / "rls_rgb"
                / f"{file_stem}_rls_rgb_{method}.png"
            )

            save_image(rls_rgb_image, rls_output)

        print(f"[DONE] {condition}: {csv_file.name}")


if __name__ == "__main__":
    paa_root = Path("results") / "paa"
    image_root = Path("results") / "images"

    conditions = [
        "normal",
        "overcharge",
        "overdischarge",
    ]

    methods = [
        "gasf",
        "mtf",
    ]

    image_size = 50

    for condition_name in conditions:
        process_paa_condition(
            condition=condition_name,
            paa_root=paa_root,
            image_root=image_root,
            methods=methods,
            image_size=image_size,
        )

    print("GASF/MTF image encoding from PAA-preprocessed data completed.")