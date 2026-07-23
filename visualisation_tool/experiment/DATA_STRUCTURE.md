# Real-Life Data Structure Guide

This guide explains how the real-life data is organized in `@data/real_life/` and how to write Python scripts to load and visualize it.

## Directory Structure

```
data/real_life/
└── <TIMESTAMP>/              # Timestamped experiment run (e.g., 2026-07-22_19-32-47)
    ├── infer_log.zarr/       # Inference/model output logs
    └── online_data.zarr/     # Online learning episode data
```

Each experiment run is stored in a directory named with a timestamp (format: `YYYY-MM-DD_HH-MM-SS`). Within each run, there are two Zarr files containing different types of data.

## Data Format: Zarr

Data is stored in **Zarr** format, a chunked array storage format that's efficient for reading large datasets. Load Zarr files using the `zarr` Python library:

```python
import zarr
data = zarr.open(path_to_zarr, mode='r')
```

## infer_log.zarr

Contains timestamped inference logs from the model during the experiment.

### Arrays

| Array Name | Description | Typical Shape |
|-----------|-------------|---------------|
| `ground_truth` | Ground truth labels/targets | (N, ...) |
| `output` | Model output predictions | (N, ...) |
| `timestamp_ns` | Nanosecond timestamps | (N, 2) or (N,) |
| `generation_id` | Generation/iteration ID | (N,) |
| `obs` | Observations used by the model | (N, ...) |
| `state_arrival_ns` | State message arrival timestamps | (N,) |
| `accel_arrival_ns` | Acceleration data arrival timestamps | (N,) |
| `setpoint_arrival_ns` | Setpoint data arrival timestamps | (N,) |
| `johnny_arrival_ns` | Johnny module arrival timestamps | (N,) |
| `swap_events` | Swap event markers | (N,) |

### Loading infer_log data

```python
import zarr

infer_log = zarr.open('data/real_life/<TIMESTAMP>/infer_log.zarr', mode='r')

# Load full arrays into memory
ground_truth = infer_log['ground_truth'][:]
outputs = infer_log['output'][:]
timestamps = infer_log['timestamp_ns'][:]

# Or access specific slices without loading everything
first_100 = infer_log['ground_truth'][:100]
```

## online_data.zarr

Contains data organized by episodes during online learning phases.

### Arrays

| Array Name | Description |
|-----------|-------------|
| `episodes` | Episode boundaries or metadata | 
| `episode_timestamps` | Timestamps for each episode | 
| `episode_generation_id` | Generation ID per episode |
| `episode_*_arrival_ns` | Per-episode arrival timestamps (state, accel, setpoint, johnny) |

### Loading online_data

```python
online = zarr.open('data/real_life/<TIMESTAMP>/online_data.zarr', mode='r')

# Access episode data
episodes = online['episodes'][:]
episode_timestamps = online['episode_timestamps'][:]
```

## Quick Start Example

```python
import zarr
import numpy as np
from pathlib import Path

# Set data directory
data_dir = Path('data/real_life/2026-07-22_19-32-47')

# Load inference logs
infer_log = zarr.open(data_dir / 'infer_log.zarr', mode='r')
print("Arrays in infer_log:", list(infer_log.array_keys()))

# Access data
ground_truth = infer_log['ground_truth'][:]
outputs = infer_log['output'][:]

# Inspect shapes
print(f"Ground truth shape: {ground_truth.shape}")
print(f"Outputs shape: {outputs.shape}")

# For visualization, compute metrics
mse = np.mean((ground_truth - outputs) ** 2)
print(f"Mean Squared Error: {mse}")
```

## Common Patterns for Visualization

### 1. Time-series plotting
```python
import matplotlib.pyplot as plt

timestamps = infer_log['timestamp_ns'][:] / 1e9  # Convert to seconds
outputs = infer_log['output'][:]

plt.plot(timestamps, outputs)
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.show()
```

### 2. Comparing predictions vs ground truth
```python
ground_truth = infer_log['ground_truth'][:]
outputs = infer_log['output'][:]

plt.figure(figsize=(12, 5))
plt.plot(ground_truth, label='Ground Truth', alpha=0.7)
plt.plot(outputs, label='Model Output', alpha=0.7)
plt.legend()
plt.show()
```

### 3. Computing error over time
```python
errors = np.abs(ground_truth - outputs)
plt.plot(errors)
plt.ylabel('Absolute Error')
plt.xlabel('Sample Index')
plt.show()
```

## Tips for Writing Visualization Scripts

1. **Handle large datasets**: Use Zarr slicing to load only the data you need, not entire arrays
   ```python
   subset = infer_log['output'][1000:2000]  # Only load this slice
   ```

2. **Check array shapes and types before plotting**:
   ```python
   print(infer_log['ground_truth'].shape)
   print(infer_log['ground_truth'].dtype)
   ```

3. **Use timestamps for x-axis**: Most arrays have corresponding `*_arrival_ns` or `timestamp_ns` timestamps

4. **Explore structure programmatically**:
   ```python
   for key in infer_log.array_keys():
       arr = infer_log[key]
       print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
   ```

## File Location Notes

- Data path: `/home/nodale/Thesis/Thesis-Invizible/data/real_life/`
- Or relative to repo root: `data/real_life/`
- Each experiment has a unique timestamp directory within `real_life/`
