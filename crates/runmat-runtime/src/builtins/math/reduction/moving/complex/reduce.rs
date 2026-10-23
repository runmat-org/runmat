//! Native component arithmetic, preserving the existing reduction order.
use super::order::{
    compare_complex_auto, complex_median, complex_median_with_fill, complex_pow_usize,
};
use super::variance::{variance_complex, variance_complex_with_fill};
use super::*;

pub(super) fn reduce_complex_window<T: Float>(
    op: MovingOp,
    values: &[(T, T)],
    saw_nan: bool,
    normalization: VarianceNormalization,
) -> (T, T) {
    if saw_nan {
        return (T::nan(), T::nan());
    }
    if values.is_empty() {
        return match op {
            MovingOp::Sum => (T::zero(), T::zero()),
            MovingOp::Prod => (T::one(), T::zero()),
            MovingOp::Mean
            | MovingOp::Min
            | MovingOp::Max
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => (T::nan(), T::nan()),
        };
    }
    match op {
        MovingOp::Sum => values.iter().fold((T::zero(), T::zero()), |acc, value| {
            (acc.0 + value.0, acc.1 + value.1)
        }),
        MovingOp::Mean => {
            let sum = reduce_complex_window(MovingOp::Sum, values, false, normalization);
            (
                sum.0 / number::<T>(values.len()),
                sum.1 / number::<T>(values.len()),
            )
        }
        MovingOp::Prod => values.iter().fold((T::one(), T::zero()), |acc, value| {
            (
                acc.0 * value.0 - acc.1 * value.1,
                acc.0 * value.1 + acc.1 * value.0,
            )
        }),
        MovingOp::Min => values
            .iter()
            .copied()
            .min_by(compare_complex_auto)
            .unwrap_or((T::nan(), T::nan())),
        MovingOp::Max => values
            .iter()
            .copied()
            .max_by(compare_complex_auto)
            .unwrap_or((T::nan(), T::nan())),
        MovingOp::Median => complex_median(values),
        MovingOp::Var => (variance_complex(values, normalization), T::zero()),
        MovingOp::Std => (variance_complex(values, normalization).sqrt(), T::zero()),
    }
}

pub(super) fn reduce_complex_window_with_fill<T: Float>(
    op: MovingOp,
    values: &[(T, T)],
    saw_nan: bool,
    fill: Option<((T, T), usize)>,
    nan_mode: NanMode,
    normalization: VarianceNormalization,
) -> (T, T) {
    let Some((fill, fill_count)) = fill else {
        return reduce_complex_window(op, values, saw_nan, normalization);
    };
    if fill_count == 0 {
        return reduce_complex_window(op, values, saw_nan, normalization);
    }
    let fill_is_nan = fill.0.is_nan() || fill.1.is_nan();
    if saw_nan || (fill_is_nan && nan_mode == NanMode::Include) {
        return (T::nan(), T::nan());
    }
    if fill_is_nan && nan_mode == NanMode::Omit {
        return reduce_complex_window(op, values, false, normalization);
    }
    match op {
        MovingOp::Sum => {
            let sum = reduce_complex_window(MovingOp::Sum, values, false, normalization);
            (
                sum.0 + fill.0 * number::<T>(fill_count),
                sum.1 + fill.1 * number::<T>(fill_count),
            )
        }
        MovingOp::Mean => {
            let sum = reduce_complex_window_with_fill(
                MovingOp::Sum,
                values,
                false,
                Some((fill, fill_count)),
                nan_mode,
                normalization,
            );
            (
                sum.0 / number::<T>(values.len() + fill_count),
                sum.1 / number::<T>(values.len() + fill_count),
            )
        }
        MovingOp::Prod => {
            let product = reduce_complex_window(MovingOp::Prod, values, false, normalization);
            let fill_product = complex_pow_usize(fill, fill_count);
            (
                product.0 * fill_product.0 - product.1 * fill_product.1,
                product.0 * fill_product.1 + product.1 * fill_product.0,
            )
        }
        MovingOp::Min => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .min_by(compare_complex_auto)
            .unwrap_or((T::nan(), T::nan())),
        MovingOp::Max => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .max_by(compare_complex_auto)
            .unwrap_or((T::nan(), T::nan())),
        MovingOp::Median => complex_median_with_fill(values, fill, fill_count),
        MovingOp::Var => (
            variance_complex_with_fill(values, fill, fill_count, normalization),
            T::zero(),
        ),
        MovingOp::Std => (
            variance_complex_with_fill(values, fill, fill_count, normalization).sqrt(),
            T::zero(),
        ),
    }
}
