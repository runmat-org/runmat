//! Exact power-of-two factor balance and conservative stage admission.

use num_complex::Complex;
use std::borrow::Cow;

use super::{conv2_allocation_error, Matrix};
use crate::builtins::math::signal::common::{SignalCooperation, SIGNAL_WORK_CHUNK};
use crate::BuiltinResult;

pub(super) struct BalancedFactors<'a> {
    pub(super) column: Cow<'a, [Complex<f64>]>,
    pub(super) row: Cow<'a, [Complex<f64>]>,
}

#[derive(Clone, Copy)]
struct ComponentRange {
    maximum: f64,
    minimum_nonzero: f64,
}

/// Balance factors by an exactly reversible binary power and admit the two
/// passes only when conservative class-aware intermediate bounds are safe.
/// Unrepresentable scaling or potentially unsafe intermediates use the
/// original general-kernel route instead.
pub(super) async fn balanced_factors<'a>(
    signal: &Matrix,
    column: &'a [Complex<f64>],
    row: &'a [Complex<f64>],
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<BalancedFactors<'a>>> {
    let Some(signal_range) = component_range(&signal.data, cooperation).await? else {
        return Ok(None);
    };
    let Some(column_range) = component_range(column, cooperation).await? else {
        return Ok(None);
    };
    let Some(_) = component_range(row, cooperation).await? else {
        return Ok(None);
    };
    let exponent = if column_range.maximum == 0.0 {
        0
    } else {
        let (fraction, exponent) = libm::frexp(column_range.maximum);
        exponent - if fraction == 0.5 { 1 } else { 0 }
    };
    let factors = if exponent == 0 {
        BalancedFactors {
            column: Cow::Borrowed(column),
            row: Cow::Borrowed(row),
        }
    } else {
        let mut scaled_column = Vec::new();
        scaled_column
            .try_reserve_exact(column.len())
            .map_err(conv2_allocation_error)?;
        let mut scaled_row = Vec::new();
        scaled_row
            .try_reserve_exact(row.len())
            .map_err(conv2_allocation_error)?;
        let mut work = 0_u64;
        for source in column {
            let scaled = Complex::new(
                libm::ldexp(source.re, -exponent),
                libm::ldexp(source.im, -exponent),
            );
            if loses_component(*source, scaled, exponent, single_output) {
                return Ok(None);
            }
            scaled_column.push(scaled);
            work += 1;
            if work == SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
        cooperation.completed(work).await?;
        work = 0;
        for source in row {
            let scaled = Complex::new(
                libm::ldexp(source.re, exponent),
                libm::ldexp(source.im, exponent),
            );
            if loses_component(*source, scaled, -exponent, single_output) {
                return Ok(None);
            }
            scaled_row.push(scaled);
            work += 1;
            if work == SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
        cooperation.completed(work).await?;
        BalancedFactors {
            column: Cow::Owned(scaled_column),
            row: Cow::Owned(scaled_row),
        }
    };
    let Some(column_range) = component_range(factors.column.as_ref(), cooperation).await? else {
        return Ok(None);
    };
    let Some(row_range) = component_range(factors.row.as_ref(), cooperation).await? else {
        return Ok(None);
    };
    let precision_limit = if single_output {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let precision_normal = if single_output {
        f64::from(f32::MIN_POSITIVE)
    } else {
        f64::MIN_POSITIVE
    };
    if may_underflow_first_stage(signal_range, column_range, precision_normal) {
        return Ok(None);
    }
    let Some(vertical_bound) = upper_stage_bound(
        signal_range.maximum,
        column_range.maximum,
        signal.rows.min(column.len()),
        precision_limit,
    ) else {
        return Ok(None);
    };
    if upper_stage_bound(
        vertical_bound,
        row_range.maximum,
        signal.cols.min(row.len()),
        precision_limit,
    )
    .is_none()
    {
        return Ok(None);
    }
    Ok(Some(factors))
}

async fn component_range(
    values: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<ComponentRange>> {
    let mut range = ComponentRange {
        maximum: 0.0,
        minimum_nonzero: f64::INFINITY,
    };
    for chunk in values.chunks(SIGNAL_WORK_CHUNK as usize) {
        for value in chunk {
            for component in [value.re, value.im] {
                if !component.is_finite() {
                    return Ok(None);
                }
                let magnitude = component.abs();
                range.maximum = range.maximum.max(magnitude);
                if magnitude != 0.0 {
                    range.minimum_nonzero = range.minimum_nonzero.min(magnitude);
                }
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Some(range))
}

fn may_underflow_first_stage(
    signal: ComponentRange,
    column: ComponentRange,
    normal_minimum: f64,
) -> bool {
    signal.minimum_nonzero.is_finite()
        && column.minimum_nonzero.is_finite()
        && signal.minimum_nonzero < normal_minimum / column.minimum_nonzero
}

/// Four component products per complex accumulation is a conservative bound.
/// If even this bound is unsafe, use the established general-kernel route.
fn upper_stage_bound(input: f64, kernel: f64, terms: usize, limit: f64) -> Option<f64> {
    if input == 0.0 || kernel == 0.0 {
        return Some(0.0);
    }
    let work = 4.0 * kernel * terms as f64;
    if !work.is_finite() || input > limit / work {
        return None;
    }
    let bound = input * work;
    (bound.is_finite() && bound <= limit).then_some(bound)
}

fn loses_component(
    source: Complex<f64>,
    scaled: Complex<f64>,
    restore_exponent: i32,
    single_output: bool,
) -> bool {
    [(source.re, scaled.re), (source.im, scaled.im)]
        .iter()
        .any(|(source, scaled)| {
            if !scaled.is_finite()
                || (source != &0.0 && scaled == &0.0)
                || libm::ldexp(*scaled, restore_exponent).to_bits() != source.to_bits()
            {
                return true;
            }
            if !single_output {
                return false;
            }
            let original = *source as f32;
            if !original.is_finite() || original == 0.0 {
                return false;
            }
            let balanced = *scaled as f32;
            !balanced.is_finite()
                || balanced == 0.0
                || (libm::ldexp(f64::from(balanced), restore_exponent) as f32).to_bits()
                    != original.to_bits()
        })
}
