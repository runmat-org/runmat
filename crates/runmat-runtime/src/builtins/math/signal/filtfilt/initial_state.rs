//! The one initial-state system used by host and resident filtfilt routes.
use num_complex::Complex;

use super::{
    filtfilt_error_with_detail, NormalizedCoefficients, BUILTIN_NAME, FILTFILT_ERROR_INTERNAL,
    FILTFILT_ERROR_INVALID_COEFFICIENTS,
};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use std::collections::TryReserveError;

fn allocation_error(source: TryReserveError) -> crate::RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: initial-state allocation failed",
        FILTFILT_ERROR_INTERNAL.message
    ))
    .with_builtin(BUILTIN_NAME)
    .with_source(source);
    if let Some(identifier) = FILTFILT_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_gpu_gather_retry(GpuGatherRetry::Never).build()
}

fn finite(value: Complex<f64>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn range_error() -> crate::RuntimeError {
    filtfilt_error_with_detail(
        &FILTFILT_ERROR_INVALID_COEFFICIENTS,
        "filter initial-state system exceeds the floating-point range",
    )
}

fn singular_error() -> crate::RuntimeError {
    filtfilt_error_with_detail(
        &FILTFILT_ERROR_INVALID_COEFFICIENTS,
        "filter initial-state system is singular",
    )
}

const SOLVE_CHUNK: u64 = 4096;

pub(super) async fn lfilter_zi(
    coeffs: &NormalizedCoefficients,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    cooperation.completed(0).await?;
    let n = coeffs.state_len;
    if n == 0 {
        return Ok(Vec::new());
    }
    n.checked_mul(n)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<Complex<f64>>()))
        .ok_or_else(|| {
            let mut error =
                filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, "initial-state size overflow");
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })?;
    let mut matrix = Vec::new();
    matrix.try_reserve_exact(n).map_err(allocation_error)?;
    for _ in 0..n {
        let mut row = Vec::new();
        row.try_reserve_exact(n).map_err(allocation_error)?;
        row.resize(n, Complex::new(0.0, 0.0));
        matrix.push(row);
        cooperation
            .completed(u64::try_from(n).unwrap_or(u64::MAX))
            .await?;
    }
    for (row, row_values) in matrix.iter_mut().enumerate() {
        for (col, value) in row_values.iter_mut().enumerate() {
            if row == col {
                *value += Complex::new(1.0, 0.0);
            }
            let a_t = if col == 0 {
                -coeffs.a[row + 1]
            } else if row + 1 == col {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            *value -= a_t;
        }
        cooperation
            .completed(u64::try_from(n).unwrap_or(u64::MAX))
            .await?;
    }
    let mut rhs = Vec::new();
    rhs.try_reserve_exact(n).map_err(allocation_error)?;
    for idx in 0..n {
        rhs.push(coeffs.b[idx + 1] - coeffs.a[idx + 1] * coeffs.b[0]);
    }
    cooperation
        .completed(u64::try_from(n).unwrap_or(u64::MAX))
        .await?;
    solve_linear(matrix, rhs, cooperation).await
}

async fn solve_linear(
    mut matrix: Vec<Vec<Complex<f64>>>,
    mut rhs: Vec<Complex<f64>>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    cooperation.completed(0).await?;
    let n = rhs.len();
    for (row, values) in matrix.iter_mut().enumerate() {
        let scale = values.iter().fold(0.0_f64, |largest, value| {
            largest.max(value.re.abs()).max(value.im.abs())
        });
        if !scale.is_finite() || !finite(rhs[row]) {
            return Err(range_error());
        }
        if scale == 0.0 {
            return Err(singular_error());
        }
        for value in values {
            *value = Complex::new(value.re / scale, value.im / scale);
        }
        cooperation
            .completed(u64::try_from(n).unwrap_or(u64::MAX))
            .await?;
        rhs[row] = Complex::new(rhs[row].re / scale, rhs[row].im / scale);
        if !finite(rhs[row]) {
            return Err(range_error());
        }
    }

    let mut work = 0_u64;
    for pivot_idx in 0..n {
        let Some(pivot_row) = (pivot_idx..n).max_by(|&left, &right| {
            matrix[left][pivot_idx]
                .norm()
                .total_cmp(&matrix[right][pivot_idx].norm())
        }) else {
            return Err(singular_error());
        };
        if matrix[pivot_row][pivot_idx] == Complex::new(0.0, 0.0) {
            return Err(singular_error());
        }
        if pivot_row != pivot_idx {
            matrix.swap(pivot_row, pivot_idx);
            rhs.swap(pivot_row, pivot_idx);
        }
        let pivot = matrix[pivot_idx][pivot_idx];
        let pivot_rhs = rhs[pivot_idx];
        let (pivot_rows, trailing_rows) = matrix.split_at_mut(pivot_idx + 1);
        let pivot_row_values = &pivot_rows[pivot_idx];
        for (offset, values) in trailing_rows.iter_mut().enumerate() {
            let row = pivot_idx + 1 + offset;
            let factor = values[pivot_idx] / pivot;
            for (col, pivot_value) in pivot_row_values.iter().enumerate().skip(pivot_idx) {
                values[col] -= factor * *pivot_value;
                if !finite(values[col]) {
                    return Err(range_error());
                }
                work += 1;
                if work == SOLVE_CHUNK {
                    cooperation.completed(work).await?;
                    work = 0;
                }
            }
            rhs[row] -= factor * pivot_rhs;
            if !finite(rhs[row]) {
                return Err(range_error());
            }
        }
    }

    cooperation.completed(work).await?;

    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(allocation_error)?;
    out.resize(n, Complex::new(0.0, 0.0));
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for (col, value) in matrix[row].iter().enumerate().skip(row + 1) {
            acc -= *value * out[col];
            work += 1;
            if work == SOLVE_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
        let pivot = matrix[row][row];
        if pivot == Complex::new(0.0, 0.0) {
            return Err(singular_error());
        }
        out[row] = acc / pivot;
        if !finite(out[row]) {
            return Err(range_error());
        }
    }
    cooperation.completed(work).await?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{cell::Cell, num::NonZeroU64, rc::Rc};

    fn solve(
        matrix: Vec<Vec<Complex<f64>>>,
        rhs: Vec<Complex<f64>>,
    ) -> BuiltinResult<Vec<Complex<f64>>> {
        futures::executor::block_on(solve_linear(
            matrix,
            rhs,
            &mut SignalCooperation::new(None, BUILTIN_NAME),
        ))
    }

    fn state(coeffs: &NormalizedCoefficients) -> BuiltinResult<Vec<Complex<f64>>> {
        futures::executor::block_on(lfilter_zi(
            coeffs,
            &mut SignalCooperation::new(None, BUILTIN_NAME),
        ))
    }

    #[test]
    fn small_nonzero_pivot_remains_admissible_without_fixed_epsilon() {
        let solution = solve(
            vec![vec![Complex::new(1.0e-14, 0.0)]],
            vec![Complex::new(1.0, 0.0)],
        )
        .expect("finite one-state solution");
        assert!((solution[0].re - 1.0e14).abs() <= 1.0);
        assert_eq!(solution[0].im, 0.0);
        let coeffs = NormalizedCoefficients::new(
            &[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            &[Complex::new(1.0, 0.0), Complex::new(-(1.0 - 1.0e-14), 0.0)],
        )
        .expect("near-unit-root coefficients");
        let state = state(&coeffs).expect("finite near-unit-root initial state");
        assert!(state[0].re.is_finite());
        assert!(state[0].re > 1.0e13);
        let complex = solve(
            vec![vec![Complex::new(1.0e-14, 1.0e-14)]],
            vec![Complex::new(1.0, 1.0)],
        )
        .expect("finite complex initial state");
        assert!((complex[0].re - 1.0e14).abs() <= 1.0);
        assert!(complex[0].im.abs() <= 1.0);
    }

    #[test]
    fn zero_and_nonrepresentable_systems_reject_without_publishing_nan() {
        assert!(solve(
            vec![vec![Complex::new(0.0, 0.0)]],
            vec![Complex::new(1.0, 0.0)],
        )
        .is_err());
        assert!(solve(
            vec![vec![Complex::new(f64::MIN_POSITIVE, 0.0)]],
            vec![Complex::new(f64::MAX, 0.0)],
        )
        .is_err());
    }

    #[test]
    fn multistate_scaled_pivoting_reconstructs_independent_real_and_complex_solutions() {
        let real = solve(
            vec![
                vec![Complex::new(1.0e-14, 0.0), Complex::new(1.0, 0.0)],
                vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            ],
            vec![Complex::new(3.0 + 2.0e-14, 0.0), Complex::new(5.0, 0.0)],
        )
        .expect("two-state real solution");
        assert!((real[0].re - 2.0).abs() < 1.0e-10);
        assert!((real[1].re - 3.0).abs() < 1.0e-10);

        let complex = solve(
            vec![
                vec![Complex::new(1.0e-14, 1.0e-14), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(2.0, -1.0)],
            ],
            vec![Complex::new(1.0, 1.0), Complex::new(3.0, 1.0)],
        )
        .expect("two-state complex solution");
        assert!((complex[0].re - 1.0e14).abs() <= 1.0);
        assert!(complex[0].im.abs() <= 1.0);
        assert!((complex[1].re - 1.0).abs() < 1.0e-12);
        assert!((complex[1].im - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn quadratic_workspace_overflow_is_a_structured_internal_error() {
        let impossible = NormalizedCoefficients {
            b: Vec::new(),
            a: Vec::new(),
            order: usize::MAX,
            state_len: usize::MAX,
        };
        let error = state(&impossible).expect_err("checked square size before indexing");
        assert_eq!(error.identifier(), FILTFILT_ERROR_INTERNAL.identifier);
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    }

    #[test]
    fn initial_state_solver_observes_invocation_cancellation_after_work() {
        let coeffs = NormalizedCoefficients::new(
            &[
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
            &[
                Complex::new(1.0, 0.0),
                Complex::new(-0.5, 0.0),
                Complex::new(0.2, 0.0),
            ],
        )
        .unwrap();
        let source = Rc::new(CancellationSource::new());
        let turns = Rc::new(Cell::new(0));
        let cancelled = source.clone();
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(9).unwrap()),
            source.token(),
            move || {
                observed.set(observed.get() + 1);
                cancelled.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let error = futures::executor::block_on(lfilter_zi(
            &coeffs,
            &mut SignalCooperation::with_budget(budget, BUILTIN_NAME),
        ))
        .expect_err("cancelled invocation must not publish initial state");
        assert_eq!(turns.get(), 1);
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    }
}
