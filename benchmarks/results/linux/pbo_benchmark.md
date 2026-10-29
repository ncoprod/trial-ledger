Machine: Linux-7.0.0-1017-azure-x86_64-with-glibc2.43, 2 logical CPUs, Python 3.13.18, NumPy 2.5.4; pypbo @ 4d723f0.

Timing: median wall-clock runs; all samples recorded, shared machine.

| T | N | S | splits | pypbo | NumPy generic | NumPy block-stats | Rust 1 thread | Rust parallel | Rust ∥ vs pypbo | Rust 1t vs NumPy block-stats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 50 | 8 | 70 | 0.04 s | 0.01 s | 4.3 ms | 0.13 ms | 0.12 ms | 371x | 32.6x |
| 1000 | 10 | 16 | 12870 | 3.88 s | 1.92 s | 751.7 ms | 3.72 ms | 3.42 ms | 1,136x | 202.0x |
| 2190 | 100 | 10 | 252 | 0.24 s | 0.13 s | 15.5 ms | 0.61 ms | 0.59 ms | 405x | 25.3x |
| 2190 | 500 | 8 | 70 | 0.24 s | 0.13 s | 6.4 ms | 1.34 ms | 1.34 ms | 182x | 4.8x |
| 2190 | 1000 | 16 | 12870 | not run (~224 GB) | 54.36 s | 1189.9 ms | 192.27 ms | 189.39 ms | n/a | 6.2x |
