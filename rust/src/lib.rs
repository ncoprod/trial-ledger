//! Rust core for trial-ledger: Probability of Backtest Overfitting via combinatorially
//! symmetric cross-validation (CSCV, Bailey, Borwein, López de Prado & Zhu 2017).
//!
//! The (T x N) returns matrix is cut into S contiguous blocks of equal length (leading
//! residual rows are dropped, as in pypbo). Per-block sums and sums of squares are computed
//! once; each of the C(S, S/2) train/test splits then needs only O(S * N) work to get every
//! trial's Sharpe ratio on both halves (the test half is the complement of the train half).
//! Splits are processed in parallel with rayon and returned in `itertools.combinations` order.

use rayon::prelude::*;

/// All size-k combinations of 0..n in lexicographic order (same order as itertools).
pub fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // rightmost position that can still be incremented
        let mut i = k as isize - 1;
        while i >= 0 && idx[i as usize] == i as usize + n - k {
            i -= 1;
        }
        if i < 0 {
            return out;
        }
        let i = i as usize;
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[inline]
fn sharpe(sum: f64, sumsq: f64, n: f64) -> f64 {
    let mean = sum / n;
    let var = (sumsq - n * mean * mean) / (n - 1.0);
    let sd = if var > 0.0 { var.sqrt() } else { 0.0 };
    if sd > 0.0 {
        mean / sd
    } else {
        0.0
    }
}

/// Result of one CSCV split: (logit, IS Sharpe of the selected trial, its OOS Sharpe).
pub type Split = (f64, f64, f64);

/// Pure-Rust CSCV on a row-major (T x N) matrix.
pub fn pbo_cscv(data: &[f64], t: usize, n: usize, s: usize) -> Result<Vec<Split>, String> {
    if s < 2 || s % 2 != 0 {
        return Err("n_partitions must be an even integer >= 2".into());
    }
    if n < 2 {
        return Err("need at least 2 trials (columns)".into());
    }
    if data.len() != t * n {
        return Err("data length does not match shape".into());
    }
    let size = t / s;
    if size < 2 {
        return Err("each partition needs at least 2 rows".into());
    }
    let offset = t % s;

    // per-block sufficient statistics, block-major: bsum[b * n + j]
    let mut bsum = vec![0.0f64; s * n];
    let mut bsq = vec![0.0f64; s * n];
    bsum.par_chunks_mut(n)
        .zip(bsq.par_chunks_mut(n))
        .enumerate()
        .for_each(|(b, (sum_b, sq_b))| {
            let start = offset + b * size;
            for row in start..start + size {
                let r = &data[row * n..(row + 1) * n];
                for j in 0..n {
                    let x = r[j];
                    sum_b[j] += x;
                    sq_b[j] += x * x;
                }
            }
        });
    let mut tot_sum = vec![0.0f64; n];
    let mut tot_sq = vec![0.0f64; n];
    for b in 0..s {
        for j in 0..n {
            tot_sum[j] += bsum[b * n + j];
            tot_sq[j] += bsq[b * n + j];
        }
    }

    let half = (s / 2 * size) as f64;
    let splits = combinations(s, s / 2);
    let out: Vec<Split> = splits
        .par_iter()
        .map_init(
            || (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]),
            |(tr_sum, tr_sq, te_sr), train| {
                tr_sum.iter_mut().for_each(|x| *x = 0.0);
                tr_sq.iter_mut().for_each(|x| *x = 0.0);
                for &b in train {
                    let bs = &bsum[b * n..(b + 1) * n];
                    let bq = &bsq[b * n..(b + 1) * n];
                    for j in 0..n {
                        tr_sum[j] += bs[j];
                        tr_sq[j] += bq[j];
                    }
                }
                // IS-best trial (first maximum, like numpy.argmax)
                let mut best = 0usize;
                let mut best_sr = f64::NEG_INFINITY;
                for j in 0..n {
                    let sr = sharpe(tr_sum[j], tr_sq[j], half);
                    if sr > best_sr {
                        best_sr = sr;
                        best = j;
                    }
                }
                for j in 0..n {
                    te_sr[j] = sharpe(tot_sum[j] - tr_sum[j], tot_sq[j] - tr_sq[j], half);
                }
                let v = te_sr[best];
                let mut less = 0usize;
                let mut equal = 0usize;
                for &x in te_sr.iter() {
                    if x < v {
                        less += 1;
                    } else if x == v {
                        equal += 1;
                    }
                }
                let rank = less as f64 + 0.5 * (equal as f64 - 1.0) + 1.0;
                let w = rank / (n as f64 + 1.0);
                ((w / (1.0 - w)).ln(), best_sr, v)
            },
        )
        .collect();
    Ok(out)
}

#[cfg(feature = "python")]
mod python {
    use super::pbo_cscv;
    use numpy::{PyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// pbo_cscv_sharpe(returns: float64[T, N], n_partitions: int)
    ///     -> (logits, is_sharpe_of_best, oos_sharpe_of_best), each float64[C(S, S/2)]
    #[pyfunction]
    fn pbo_cscv_sharpe<'py>(
        py: Python<'py>,
        returns: PyReadonlyArray2<'py, f64>,
        n_partitions: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let arr = returns.as_array();
        let (t, n) = (arr.nrows(), arr.ncols());
        let data: Vec<f64> = match arr.as_slice() {
            Some(s) => s.to_vec(),
            None => arr.iter().copied().collect(),
        };
        let res = py
            .detach(|| pbo_cscv(&data, t, n, n_partitions))
            .map_err(PyValueError::new_err)?;
        let logits: Vec<f64> = res.iter().map(|r| r.0).collect();
        let is_best: Vec<f64> = res.iter().map(|r| r.1).collect();
        let oos: Vec<f64> = res.iter().map(|r| r.2).collect();
        Ok((
            PyArray1::from_vec(py, logits),
            PyArray1::from_vec(py, is_best),
            PyArray1::from_vec(py, oos),
        ))
    }

    /// Number of rayon worker threads in use.
    #[pyfunction]
    fn num_threads() -> usize {
        rayon::current_num_threads()
    }

    #[pymodule]
    fn trial_ledger_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(pbo_cscv_sharpe, m)?)?;
        m.add_function(wrap_pyfunction!(num_threads, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_count_and_order() {
        let c = combinations(6, 3);
        assert_eq!(c.len(), 20);
        assert_eq!(c[0], vec![0, 1, 2]);
        assert_eq!(c[1], vec![0, 1, 3]);
        assert_eq!(c[19], vec![3, 4, 5]);
        assert_eq!(combinations(16, 8).len(), 12870);
    }

    #[test]
    fn pbo_rejects_bad_input() {
        assert!(pbo_cscv(&[0.0; 20], 10, 2, 3).is_err());
        assert!(pbo_cscv(&[0.0; 10], 10, 1, 2).is_err());
    }

    #[test]
    fn dominant_trial_is_never_overfit() {
        // trial 0 has a large constant edge; it is best IS and OOS in every split
        let (t, n) = (64usize, 4usize);
        let mut data = vec![0.0; t * n];
        for row in 0..t {
            for j in 0..n {
                let noise = (((row * 7 + j * 13) % 11) as f64 - 5.0) * 0.01;
                data[row * n + j] = noise + if j == 0 { 0.5 } else { 0.0 };
            }
        }
        let res = pbo_cscv(&data, t, n, 8).unwrap();
        assert_eq!(res.len(), 70);
        assert!(res.iter().all(|r| r.0 > 0.0));
    }
}
