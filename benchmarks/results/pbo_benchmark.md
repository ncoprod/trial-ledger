Machine: macOS-27.0-arm64-arm-64bit-Mach-O, 8 logical CPUs, Python 3.13.17, NumPy 2.5.4; pypbo @ 4d723f0.

Timing: median wall-clock runs; all samples recorded, shared machine.

| T | N | S | splits | pypbo | NumPy generic | NumPy block-stats | Rust 1 thread | Rust parallel | Rust ∥ vs pypbo | Rust 1t vs NumPy block-stats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 50 | 8 | 70 | 0.25 s | 0.02 s | 5.1 ms | 0.91 ms | 0.25 ms | 978x | 5.7x |
| 1000 | 10 | 16 | 12870 | 72.10 s | 1.27 s | 604.1 ms | 4.00 ms | 1.56 ms | 46,273x | 150.9x |
| 2190 | 100 | 10 | 252 | 0.35 s | 0.13 s | 10.7 ms | 0.48 ms | 0.91 ms | 390x | 22.5x |
| 2190 | 500 | 8 | 70 | 0.20 s | 0.14 s | 4.7 ms | 1.96 ms | 1.92 ms | 107x | 2.4x |
| 2190 | 1000 | 16 | 12870 | not run (~224 GB) | 140.35 s | 1027.3 ms | 147.18 ms | 45.06 ms | n/a | 7.0x |
