"""
PAA preprocessing for battery voltage and RLS-derived parameter time-series.

This script resamples variable-length SOC-window sequences to a fixed target
length using Piecewise Aggregate Approximation (PAA).

In the manuscript, the target length N is set to 50.
"""

import os
import numpy as np
import pandas as pd


def paa_transform(sequence, target_length=50):
    """
    Apply Piecewise Aggregate Approximation (PAA) to a 1D time-series.

    Parameters
    ----------
    sequence : array-like
        Input time-series with arbitrary length.
    target_length : int
        Target length after PAA. In the manuscript, target_length = 50.

    Returns
    -------
    np.ndarray
        PAA-compressed time-series with length equal to target_length.
    """
    sequence = np.asarray(sequence, dtype=float)

    if sequence.ndim != 1:
        raise ValueError("Input sequence must be one-dimensional.")

    original_length = len(sequence)

    if original_length < target_length:
        raise ValueError(
            "Input sequence length must be greater than or equal to target_length."
        )

    segment_edges = np.linspace(0, original_length, target_length + 1)
    paa_sequence = np.zeros(target_length)

    for i in range(target_length):
        start = int(np.floor(segment_edges[i]))
        end = int(np.floor(segment_edges[i + 1]))

        if end <= start:
            end = start + 1

        paa_sequence[i] = np.mean(sequence[start:end])

    return paa_sequence


def paa_dataframe(df, columns=None, target_length=50):
    """
    Apply PAA to selected voltage and RLS-derived parameter columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing voltage and/or RLS-derived parameters.
    columns : list of str, optional
        Column names to be transformed by PAA.
        Default: ["Voltage", "Ri", "Rdiff", "Cdiff"].
    target_length : int
        Target PAA length. In the manuscript, target_length = 50.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing PAA-transformed columns.
    """
    if columns is None:
        columns = ["Voltage", "Ri", "Rdiff", "Cdiff"]

    output = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Column not found: {col}. "
                f"Available columns are: {list(df.columns)}"
            )

        output[col] = paa_transform(df[col].values, target_length)

    return pd.DataFrame(output)


if __name__ == "__main__":
    # Example usage with a user-provided SOC-window CSV file.
    # The experimental dataset used in the manuscript is not included in this repository.
    # Replace this path with your own data file.
    input_file = os.path.join("data", "example_soc_window.csv")

    # Example columns used before GASF/MTF transformation.
    selected_columns = ["Voltage", "Ri", "Rdiff", "Cdiff"]

    target_length = 50

    data = pd.read_csv(input_file)
    paa_data = paa_dataframe(
        data,
        columns=selected_columns,
        target_length=target_length
    )

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "paa_output.csv")
    paa_data.to_csv(output_file, index=False)

    print(f"PAA preprocessing completed. Output saved to: {output_file}")