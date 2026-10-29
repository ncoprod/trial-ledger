Machine: macOS-27.0-arm64-arm-64bit-Mach-O, 8 logical CPUs, Python 3.13.17, NumPy 2.5.4; pypbo @ 4d723f0.

| T | N | S | splits | pypbo | NumPy generic | NumPy block-stats | Rust 1 thread | Rust parallel | Rust ∥ vs pypbo | Rust 1t vs NumPy block-stats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 50 | 8 | 70 | 0.02 s | 0.01 s | 2.7 ms | 0.08 ms | 0.19 ms | 95x | 34.8x |
| 1000 | 10 | 16 | 12870 | 2.72 s | 0.85 s | 375.9 ms | 2.42 ms | 2.24 ms | 1,218x | 155.1x |
| 2190 | 100 | 10 | 252 | 0.20 s | 0.63 s | 48.1 ms | 0.94 ms | 0.32 ms | 610x | 51.0x |
| 2190 | 500 | 8 | 70 | 2.60 s | 0.18 s | 5.2 ms | 0.98 ms | 2.38 ms | 1,091x | 5.3x |
| 2190 | 1000 | 16 | 12870 | not run (~224 GB) | 132.12 s | 1028.9 ms | 140.47 ms | 56.01 ms | n/a | 7.3x |
