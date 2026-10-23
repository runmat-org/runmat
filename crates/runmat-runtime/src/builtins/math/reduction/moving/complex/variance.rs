//! Native complex-distance Welford accumulation with repeated-fill merging.
use super::*;

pub(super) fn variance_denominator<T: Float>(n: usize, normalization: VarianceNormalization) -> T {
    match normalization {
        VarianceNormalization::Population => number::<T>(n),
        VarianceNormalization::Sample if n > 1 => number::<T>(n - 1),
        VarianceNormalization::Sample => number::<T>(n),
    }
}

pub(super) fn variance_complex<T: Float>(
    values: &[(T, T)],
    normalization: VarianceNormalization,
) -> T {
    if values.is_empty() {
        return T::nan();
    }
    let mut acc = ComplexVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.finish(normalization)
}

pub(super) fn variance_complex_with_fill<T: Float>(
    values: &[(T, T)],
    fill: (T, T),
    fill_count: usize,
    normalization: VarianceNormalization,
) -> T {
    let n = values.len() + fill_count;
    if n == 0 {
        return T::nan();
    }
    let mut acc = ComplexVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.merge_repeated(fill, fill_count);
    acc.finish(normalization)
}

struct ComplexVarianceAccumulator<T> {
    n: usize,
    mean: (T, T),
    m2: T,
}

impl<T: Float> ComplexVarianceAccumulator<T> {
    fn push(&mut self, value: (T, T)) {
        self.n += 1;
        let delta = (value.0 - self.mean.0, value.1 - self.mean.1);
        self.mean.0 = self.mean.0 + delta.0 / number::<T>(self.n);
        self.mean.1 = self.mean.1 + delta.1 / number::<T>(self.n);
        let delta2 = (value.0 - self.mean.0, value.1 - self.mean.1);
        self.m2 = self.m2 + delta.0 * delta2.0 + delta.1 * delta2.1;
    }

    fn merge_repeated(&mut self, value: (T, T), count: usize) {
        if count == 0 {
            return;
        }
        if self.n == 0 {
            self.n = count;
            self.mean = value;
            self.m2 = T::zero();
            return;
        }
        let total = self.n + count;
        let delta = (value.0 - self.mean.0, value.1 - self.mean.1);
        let scale = number::<T>(self.n) * number::<T>(count) / number::<T>(total);
        self.mean.0 = self.mean.0 + delta.0 * (number::<T>(count) / number::<T>(total));
        self.mean.1 = self.mean.1 + delta.1 * (number::<T>(count) / number::<T>(total));
        self.m2 = self.m2 + (delta.0 * delta.0 + delta.1 * delta.1) * scale;
        self.n = total;
    }

    fn finish(self, normalization: VarianceNormalization) -> T {
        if self.n == 0 {
            return T::nan();
        }
        self.m2 / variance_denominator(self.n, normalization)
    }
}

impl<T: Float> Default for ComplexVarianceAccumulator<T> {
    fn default() -> Self {
        Self {
            n: 0,
            mean: (T::zero(), T::zero()),
            m2: T::zero(),
        }
    }
}
