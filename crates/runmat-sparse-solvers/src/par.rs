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
    block_partials(len, block)
        .into_iter()
        .fold(zero, |acc, value| acc + value)
}

/// One value per fixed block, in block order. Folding these from zero in order is the
/// reduction; a row-partitioned solve whose rank boundaries are block multiples gathers every
/// rank's partials and folds them the same way, so its result is bitwise the single-rank one.
fn block_partials<R: Send>(
    len: usize,
    block: impl Fn(std::ops::Range<usize>) -> R + Sync,
) -> Vec<R> {
    let blocks = len.div_ceil(REDUCTION_BLOCK);
    let range = |b: usize| b * REDUCTION_BLOCK..((b + 1) * REDUCTION_BLOCK).min(len);
    if parallel(len) {
        (0..blocks)
            .into_par_iter()
            .map(|b| block(range(b)))
            .collect()
    } else {
        (0..blocks).map(|b| block(range(b))).collect()
    }
}

/// Block partials of [`dot`].
pub fn dot_partials<T: Scalar>(x: &[T], y: &[T]) -> Vec<T> {
    debug_assert_eq!(x.len(), y.len());
    block_partials(x.len(), |range| block_dot(&x[range.clone()], &y[range]))
}

/// Folds block partials (of any rank layout, in global block order) from zero.
pub fn fold_partials<T: Scalar>(partials: impl IntoIterator<Item = T>) -> T {
    partials
        .into_iter()
        .fold(T::zero(), |acc, value| acc + value)
}

/// `Σ conj(x_i) y_i` over one block with four interleaved accumulators combined as
/// `(s0 + s1) + (s2 + s3)`: a fixed order (independent of threads) that hides add latency and
/// lets the loop vectorize.
#[inline]
fn block_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = [T::zero(); 4];
    let (xc, yc) = (x.chunks_exact(4), y.chunks_exact(4));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for k in 0..4 {
            s[k] += a[k].conj() * b[k];
        }
    }
    for (k, (a, b)) in xr.iter().zip(yr).enumerate() {
        s[k] += a.conj() * *b;
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

/// `Σ |x_i|²` over one block, four interleaved accumulators (as [`block_dot`]).
#[inline]
fn block_abs2<T: Scalar>(x: &[T]) -> f64 {
    let mut s = [0.0f64; 4];
    let chunks = x.chunks_exact(4);
    let rest = chunks.remainder();
    for a in chunks {
        for k in 0..4 {
            s[k] += a[k].abs2();
        }
    }
    for (k, a) in rest.iter().enumerate() {
        s[k] += a.abs2();
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

/// Hermitian inner product `Σ conj(x_i) y_i`.
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    blocked_reduce(x.len(), T::zero(), |range| {
        block_dot(&x[range.clone()], &y[range])
    })
}

/// Hermitian inner product together with `Σ |conj(x_i) y_i|` (one pass, same block order).
pub fn dot_and_magnitude<T: Scalar>(x: &[T], y: &[T]) -> (T, f64) {
    fold_dot_magnitude_partials(dot_and_magnitude_partials(x, y))
}

/// Block partials of [`dot_and_magnitude`].
pub fn dot_and_magnitude_partials<T: Scalar>(x: &[T], y: &[T]) -> Vec<(T, f64)> {
    debug_assert_eq!(x.len(), y.len());
    block_partials(x.len(), |range| {
        // Four interleaved accumulators, combined in a fixed order (see `block_dot`).
        let (mut dot, mut magnitude) = ([T::zero(); 4], [0.0f64; 4]);
        for (offset, i) in range.enumerate() {
            let term = x[i].conj() * y[i];
            dot[offset % 4] += term;
            magnitude[offset % 4] += term.abs();
        }
        (
            (dot[0] + dot[1]) + (dot[2] + dot[3]),
            (magnitude[0] + magnitude[1]) + (magnitude[2] + magnitude[3]),
        )
    })
}

/// Folds [`dot_and_magnitude_partials`] from zero in order.
pub fn fold_dot_magnitude_partials<T: Scalar>(
    partials: impl IntoIterator<Item = (T, f64)>,
) -> (T, f64) {
    partials
        .into_iter()
        .fold((T::zero(), 0.0), |(d, m), (pd, pm)| (d + pd, m + pm))
}

/// `out[k] = Σ_i conj(V_k[i]) w[i]` for the `count` contiguous vectors `V_k` of `vectors`, in one
/// pass over `w`: fixed blocks of rows, each block summed per vector sequentially, block partials
/// summed in block order (bitwise identical for every thread count).
pub fn multi_dot<T: Scalar>(vectors: &[T], count: usize, w: &[T]) -> Vec<T> {
    fold_multi_partials(count, multi_dot_partials(vectors, count, w))
}

/// Block partials of [`multi_dot`]: one `count`-vector per block.
pub fn multi_dot_partials<T: Scalar>(vectors: &[T], count: usize, w: &[T]) -> Vec<Vec<T>> {
    let n = w.len();
    debug_assert_eq!(vectors.len(), count * n);
    block_partials(n, |range| {
        let wb = &w[range.clone()];
        (0..count)
            .map(|k| block_dot(&vectors[k * n + range.start..k * n + range.end], wb))
            .collect::<Vec<T>>()
    })
}

/// Folds [`multi_dot_partials`] (global block order) from zero, per vector.
pub fn fold_multi_partials<T: Scalar>(
    count: usize,
    partials: impl IntoIterator<Item = Vec<T>>,
) -> Vec<T> {
    let mut out = vec![T::zero(); count];
    for partial in partials {
        for (o, p) in out.iter_mut().zip(partial) {
            *o += p;
        }
    }
    out
}

/// `w += alpha Σ_k c_k V_k` in one pass over `w` (per entry, vectors in order).
pub fn multi_axpy<T: Scalar>(vectors: &[T], coefficients: &[T], w: &mut [T], alpha: f64) {
    let n = w.len();
    debug_assert_eq!(vectors.len(), coefficients.len() * n);
    let scaled: Vec<T> = coefficients.iter().map(|c| c.scale(alpha)).collect();
    for_each_chunk_mut(w, |start, chunk| {
        for (k, c) in scaled.iter().enumerate() {
            let vk = &vectors[k * n + start..k * n + start + chunk.len()];
            for (value, v) in chunk.iter_mut().zip(vk) {
                *value += *c * *v;
            }
        }
    });
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

/// Scaled sum of squares `scale² · sum_squares` (LAPACK `xLASSQ` form), combinable in a fixed order.
#[derive(Clone, Copy)]
struct ScaledSquares {
    scale: f64,
    sum_squares: f64,
}

impl ScaledSquares {
    const ZERO: Self = Self {
        scale: 0.0,
        sum_squares: 1.0,
    };

    fn add_component(&mut self, component: f64) {
        let magnitude = component.abs();
        if magnitude == 0.0 {
            return;
        }
        if !magnitude.is_finite() {
            // NaN propagates; an infinite component makes the norm infinite.
            self.scale = if magnitude.is_nan() || self.scale.is_nan() {
                f64::NAN
            } else {
                f64::INFINITY
            };
            self.sum_squares = 1.0;
            return;
        }
        if !self.scale.is_finite() {
            return;
        }
        if self.scale < magnitude {
            let ratio = self.scale / magnitude;
            self.sum_squares = 1.0 + self.sum_squares * ratio * ratio;
            self.scale = magnitude;
        } else {
            let ratio = magnitude / self.scale;
            self.sum_squares += ratio * ratio;
        }
    }

    fn norm(self) -> f64 {
        if self.scale == 0.0 || !self.scale.is_finite() {
            return self.scale;
        }
        self.scale * self.sum_squares.sqrt()
    }
}

impl std::ops::Add for ScaledSquares {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        if self.scale.is_nan() || other.scale.is_nan() {
            return Self {
                scale: f64::NAN,
                sum_squares: 1.0,
            };
        }
        if self.scale.is_infinite() || other.scale.is_infinite() {
            return Self {
                scale: f64::INFINITY,
                sum_squares: 1.0,
            };
        }
        if other.scale == 0.0 {
            return self;
        }
        if self.scale == 0.0 {
            return other;
        }
        let scale = self.scale.max(other.scale);
        let (a, b) = (self.scale / scale, other.scale / scale);
        Self {
            scale,
            sum_squares: self.sum_squares * a * a + other.sum_squares * b * b,
        }
    }
}

/// Euclidean norm. Accumulated as a scaled sum of squares, so finite vectors whose entries would
/// overflow or underflow when squared still produce their representable norm; the blocked
/// reduction order keeps the result identical for every thread count.
pub fn norm2<T: Scalar>(x: &[T]) -> f64 {
    fold_norm_partials(norm2_partials(x))
}

/// Block partials of [`norm2`] as `(scale, sum_squares)` pairs.
pub fn norm2_partials<T: Scalar>(x: &[T]) -> Vec<(f64, f64)> {
    // Below this a block's plain sum of squares may have lost digits to gradual underflow.
    const SAFE_MINIMUM: f64 = f64::MIN_POSITIVE * (1u64 << 52) as f64;
    let partials = block_partials(x.len(), |range| {
        // Fast path: the plain sum of squares is exact enough whenever it is finite and normal
        // with margin; otherwise rescan the block with explicit scaling.
        let plain = block_abs2(&x[range.clone()]);
        if plain.is_finite() && plain >= SAFE_MINIMUM {
            return ScaledSquares {
                scale: plain.sqrt(),
                sum_squares: 1.0,
            };
        }
        let mut sum = ScaledSquares::ZERO;
        for i in range {
            sum.add_component(x[i].re());
            if T::IS_COMPLEX {
                sum.add_component(x[i].im());
            }
        }
        sum
    });
    partials
        .into_iter()
        .map(|p| (p.scale, p.sum_squares))
        .collect()
}

/// Folds [`norm2_partials`] (any rank layout, global block order) into the norm.
pub fn fold_norm_partials(partials: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    partials
        .into_iter()
        .map(|(scale, sum_squares)| ScaledSquares { scale, sum_squares })
        .fold(ScaledSquares::ZERO, |acc, value| acc + value)
        .norm()
}

/// Maximum modulus.
pub fn norm_inf<T: Scalar>(x: &[T]) -> f64 {
    x.iter().fold(0.0f64, |acc, value| acc.max(value.abs()))
}

/// Ordered map over `0..len`; parallel inside an admitted pool, serial otherwise.
pub fn map_range<R: Send>(
    len: usize,
    min_len: usize,
    work: impl Fn(usize) -> R + Sync + Send,
) -> Vec<R> {
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
