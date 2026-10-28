Machine: macOS-27.0-arm64-arm-64bit-Mach-O, 8 logical CPUs, Python 3.13.17, NumPy 2.5.4; pypbo @ 4d723f0.

| T | N | S | splits | pypbo | NumPy generic | NumPy block-stats | Rust 1 thread | Rust parallel | Rust ∥ vs pypbo | Rust 1t vs NumPy block-stats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 50 | 8 | 70 | 0.02 s | 0.01 s | 2.0 ms | 0.08 ms | 0.11 ms | 160x | 25.9x |
| 1000 | 10 | 16 | 12870 | 1.98 s | 0.86 s | 369.6 ms | 2.66 ms | 1.36 ms | 1,463x | 139.0x |
| 2190 | 100 | 10 | 252 | 0.13 s | 0.09 s | 7.6 ms | 0.44 ms | 0.48 ms | 272x | 17.4x |
| 2190 | 500 | 8 | 70 | 0.19 s | 0.11 s | 3.4 ms | 0.96 ms | 0.74 ms | 260x | 3.6x |
| 2190 | 1000 | 16 | 12870 | not run (~224 GB) | 50.87 s | 675.3 ms | 151.52 ms | 34.66 ms | n/a | 4.5x |
