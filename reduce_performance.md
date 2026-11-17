# Reduction benchmark

- GPU: NVIDIA GeForce RTX 3070 (sm_86)
- N: 16,777,216 `float` elements
- Block size: 256
- Repeats: 5 after one warm-up
- Peak theoretical bandwidth: 448.064 GB/s
- **V0 relative speedup**: `reduce_v0_time / version_time`; `reduce_v0` is 1.0x
- **Bandwidth utilization**: effective logical bytes / elapsed time / peak bandwidth
- `cub::DeviceReduce::Sum` is included as a full reduction and its temporary-storage setup is excluded from timed execution.

| Function | Time (us) | V0 relative speedup | Effective BW (GB/s) | BW utilization |
|---|---:|---:|---:|---:|
| reduce_v0 | 640.410 | 1.000x | 105.609 | 23.570% |
| gpu_reduce_baesline | 27806.516 | 0.023x | 4.827 | 1.077% |
| reduce_v1 | 440.237 | 1.455x | 153.629 | 34.287% |
| reduce_v2 | 427.354 | 1.499x | 158.260 | 35.321% |
| reduce_v3 | 225.421 | 2.841x | 298.868 | 66.702% |
| reduce_v4 | 201.523 | 3.178x | 334.309 | 74.612% |
| reduce_v5 | 201.728 | 3.175x | 333.970 | 74.536% |
| reduce_v6 | 201.350 | 3.181x | 334.596 | 74.676% |
| reduce_v7 | 165.683 | 3.865x | 405.061 | 90.403% |
| cub::DeviceReduce | 165.478 | 3.870x | 405.545 | 90.510% |
