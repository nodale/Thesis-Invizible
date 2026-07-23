"""Simple example: loading and accessing zarr data."""

import zarr
import numpy as np
from pathlib import Path


def load_zarr_data(data_dir: str, zarr_file: str) -> zarr.Group | zarr.Array:
    """Load a zarr file and return the root object."""
    path = Path(data_dir) / zarr_file
    return zarr.open(path, mode='r')


def inspect_zarr(data: zarr.Group | zarr.Array) -> None:
    """Print basic info about zarr structure."""
    if isinstance(data, zarr.Group):
        keys = list(data.array_keys())
        print(f"Group with {len(keys)} arrays:")
        for key in keys:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"Array: shape={data.shape}, dtype={data.dtype}")


def main():
    data_dir = "/home/nodale/Thesis/thesis_main_data/2026-07-22_19-32-47"

    # Load infer_log data
    print("=== infer_log.zarr ===")
    infer_log = load_zarr_data(data_dir, "infer_log.zarr")
    inspect_zarr(infer_log)

    # Access specific arrays
    ground_truth = infer_log['ground_truth'][:]  # Load into memory
    output = infer_log['output'][:]
    timestamps = infer_log['timestamp_ns'][:]

    print(f"\nLoaded {len(ground_truth)} ground truth samples")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Output shape: {output.shape}")
    print(f"First timestamp: {timestamps[0, 0]}")

    # Load online_data
    print("\n=== online_data.zarr ===")
    online_data = load_zarr_data(data_dir, "online_data.zarr")
    inspect_zarr(online_data)


if __name__ == "__main__":
    main()
