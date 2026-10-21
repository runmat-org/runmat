//! Threaded vector kernels with fixed reduction orders.
//!
//! Every reduction splits its input into fixed blocks of [`REDUCTION_BLOCK`] entries, sums each
//! block sequentially, and then sums the block partials sequentially in block order. The result is
//! therefore bitwise identical for every thread count (including one), which keeps iterative
//! solver trajectories reproducible under threads.

use rayon::prelude::*;

use crate::scalar::Scalar;

/// Fixed reduction block length (entries).
pub const REDUCTION_BLOCK: usize = 4096;
/// Minimum length before vector kernels fan out.
const PARALLEL_THRESHOLD: usize = 4 * REDUCTION_BLOCK;

/// Worker threads available to the current computation: the admitted pool's size when running
/// inside one, otherwise one. This crate never creates or uses an implicit global pool.
pub fn threads() -> usize {
    runmat_cpu_execution::current_threads()
}

/// Runs `work` on an admitted pool of `threads` workers (one runs inline).
pub fn with_threads<R: Send>(threads: usize, work: impl FnOnce() -> R + Send) -> R {
    runmat_cpu_execution::with_cpu_pool(threads.max(1), work)
        .expect("admitted CPU pool for sparse solver work")
}

pub(crate) fn faer_par() -> faer::Par {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = threads();
        if threads > 1 {
            return faer::Par::rayon(threads);
        }
    }
    faer::Par::Seq
}

#[inline]
fn parallel(len: usize) -> bool {
    len >= PARALLEL_THRESHOLD && threads() > 1
}

/// Ordered partial sums over fixed blocks, then a sequential sum of partials.
#[inline]
fn blocked_reduce<R: Copy + Send + std::ops::Add<Output = R>>(
    len: usize,
    zero: R,
    block: impl Fn(std::ops::Range<usize>) -> R + Sync,
) -> R {
    let blocks = len.div_ceil(REDUCTION_BLOCK);
    let range = |b: usize| b * REDUCTION_BLOCK..((b + 1) * REDUCTION_BLOCK).min(len);
    if parallel(len) {
        let partials: Vec<R> = (0..blocks).into_par_iter().map(|b| block(range(b))).collect();
        partials.into_iter().fold(zero, |acc, value| acc + value)
    } else {
        (0..blocks).fold(zero, |acc, b| acc + block(range(b)))
    }
}

/// Hermitian inner product `Σ conj(x_i) y_i`.
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    blocked_reduce(x.len(), T::zero(), |range| {
        let mut sum = T::zero();
        for i in range {
            sum += x[i].conj() * y[i];
        }
        sum
    })
}

/// Bilinear (unconjugated) product `Σ x_i y_i`, used by complex-symmetric methods.
pub fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    blocked_reduce(x.len(), T::zero(), |range| {
        let mut sum = T::zero();
        for i in range {
            sum += x[i] * y[i];
        }
        sum
    })
}

/// Euclidean norm.
pub fn norm2<T: Scalar>(x: &[T]) -> f64 {
    blocked_reduce(x.len(), 0.0f64, |range| {
        let mut sum = 0.0;
        for i in range {
            sum += x[i].abs2();
        }
        sum
    })
    .sqrt()
}

/// Maximum modulus.
pub fn norm_inf<T: Scalar>(x: &[T]) -> f64 {
    x.iter().fold(0.0f64, |acc, value| acc.max(value.abs()))
}

/// Ordered map over `0..len`; parallel inside an admitted pool, serial otherwise.
pub fn map_range<R: Send>(len: usize, min_len: usize, work: impl Fn(usize) -> R + Sync) -> Vec<R> {
    if threads() > 1 && len >= 2 * min_len.max(1) {
        (0..len)
            .into_par_iter()
            .with_min_len(min_len.max(1))
            .map(work)
            .collect()
    } else {
        (0..len).map(work).collect()
    }
}

/// Applies `work(start, chunk)` to disjoint chunks of `y` (in parallel when worthwhile).
pub fn for_each_chunk_mut<T: Send>(y: &mut [T], work: impl Fn(usize, &mut [T]) + Sync) {
    if parallel(y.len()) {
        y.par_chunks_mut(REDUCTION_BLOCK)
            .enumerate()
            .for_each(|(block, chunk)| work(block * REDUCTION_BLOCK, chunk));
    } else {
        work(0, y);
    }
}

/// `y += alpha x`
pub fn axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    for_each_chunk_mut(y, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value += alpha * x[start + offset];
        }
    });
}

/// `y = x + beta y`
pub fn xpby<T: Scalar>(x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    for_each_chunk_mut(y, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value = x[start + offset] + beta * *value;
        }
    });
}

/// `y = alpha x + beta y`
pub fn axpby<T: Scalar>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    for_each_chunk_mut(y, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value = alpha * x[start + offset] + beta * *value;
        }
    });
}

/// `x *= alpha`
pub fn scale<T: Scalar>(alpha: T, x: &mut [T]) {
    for_each_chunk_mut(x, |_, chunk| {
        for value in chunk {
            *value *= alpha;
        }
    });
}

/// `z = x - y`
pub fn sub_into<T: Scalar>(x: &[T], y: &[T], z: &mut [T]) {
    for_each_chunk_mut(z, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value = x[start + offset] - y[start + offset];
        }
    });
}

/// Elementwise product `z_i = d_i * x_i`.
pub fn mul_into<T: Scalar>(d: &[T], x: &[T], z: &mut [T]) {
    for_each_chunk_mut(z, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value = d[start + offset] * x[start + offset];
        }
    });
}

/// Allocates a zero vector, failing gracefully instead of aborting on exhaustion.
pub fn try_zeros<T: Scalar>(len: usize) -> crate::SolverResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| crate::SolverError::OutOfMemory {
            bytes: (len as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        })?;
    values.resize(len, T::zero());
    Ok(values)
}

/// Deterministic pseudo-random vector (SplitMix64), used for Krylov/eigen start vectors so runs
/// are reproducible.
pub fn seeded_vector<T: Scalar>(len: usize, seed: u64) -> Vec<T> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..len)
        .map(|_| {
            let re = next();
            let im = if T::IS_COMPLEX { next() } else { 0.0 };
            T::from_parts(re, im)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_are_bitwise_identical_across_thread_counts() {
        let x = seeded_vector::<f64>(100_003, 1);
        let y = seeded_vector::<f64>(100_003, 2);
        let serial = dot(&x, &y);
        let serial_norm = norm2(&x);
        for threads in [2, 3, 5, 8] {
            let (value, norm) = with_threads(threads, || (dot(&x, &y), norm2(&x)));
            assert_eq!(value.to_bits(), serial.to_bits());
            assert_eq!(norm.to_bits(), serial_norm.to_bits());
        }
    }
}
