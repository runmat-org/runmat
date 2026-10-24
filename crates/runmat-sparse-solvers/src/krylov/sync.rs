//! Thin synchronous drivers over the crate's PCG and GMRES recurrences.
//!
//! They allocate the recurrence workspace, adapt a [`LinearOperator`] and an optional
//! [`Preconditioner`] to the recurrence operations, run the recurrence with ready futures and
//! translate the outcome into a [`KrylovReport`]. They add no iteration logic of their own.

use num_complex::Complex64;

use super::gmres::{
    self, GmresGeometry, GmresMode, GmresOperations, GmresParameters, PreconditionerSide,
    ResidualCheck,
};
use super::pcg::{self, PcgGeometry, PcgOperation, PcgOperations, PcgParameters, PcgWork};
use super::{
    block_on, check_dims, ApplyStatus, Control, KrylovFailure, KrylovOptions, KrylovReport,
    StopFlag,
};
use crate::control::{ProgressPhase, SolveControl};
use crate::error::{SolverError, SolverResult};
use crate::operator::{LinearOperator, Preconditioner};
use crate::par;
use crate::scalar::Scalar;

/// Zero-sized arithmetic policy of the recurrences for the crate's [`Scalar`] types.
pub struct ScalarArithmetic;

impl<T: Scalar> pcg::Arithmetic<T> for ScalarArithmetic {
    const HERMITIAN_EPSILON: f64 = T::EPSILON;
    fn zero() -> T {
        T::zero()
    }
    fn components(value: T) -> (f64, f64) {
        (value.re(), value.im())
    }
    fn add_scaled(target: &mut T, scale: f64, value: T) {
        *target = T::from_parts(
            target.re() + scale * value.re(),
            target.im() + scale * value.im(),
        );
    }
    fn subtract_scaled(target: &mut T, scale: f64, value: T) {
        *target = T::from_parts(
            target.re() - scale * value.re(),
            target.im() - scale * value.im(),
        );
    }
    fn scaled_conjugate_product(
        left: T,
        right: T,
        left_scale: f64,
        right_scale: f64,
    ) -> (f64, f64) {
        let (ar, ai) = (left.re() / left_scale, left.im() / left_scale);
        let (br, bi) = (right.re() / right_scale, right.im() / right_scale);
        (ar * br + ai * bi, ar * bi - ai * br)
    }
    fn native_dot(left: &[T], right: &[T]) -> Option<(f64, f64, f64)> {
        let (dot, magnitude) = par::dot_and_magnitude(left, right);
        // Plain sums are exact enough when finite and clear of gradual underflow.
        (dot.re().is_finite()
            && dot.im().is_finite()
            && magnitude.is_finite()
            && (magnitude == 0.0 || magnitude >= SAFE_MINIMUM))
            .then(|| (dot.re(), dot.im(), magnitude))
    }
    fn native_step(
        x: &mut [T],
        residual: &mut [T],
        direction: &[T],
        product: &[T],
        alpha: f64,
    ) -> Option<bool> {
        let progressed = std::sync::atomic::AtomicBool::new(false);
        par::for_each_chunk_mut(x, |start, chunk| {
            let mut changed = false;
            for (offset, value) in chunk.iter_mut().enumerate() {
                let previous = *value;
                <Self as pcg::Arithmetic<T>>::add_scaled(value, alpha, direction[start + offset]);
                changed |= *value != previous;
            }
            if changed {
                progressed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        par::for_each_chunk_mut(residual, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                <Self as pcg::Arithmetic<T>>::subtract_scaled(
                    value,
                    alpha,
                    product[start + offset],
                );
            }
        });
        Some(progressed.into_inner())
    }
    fn native_direction(direction: &mut [T], source: &[T], beta: f64) -> bool {
        par::for_each_chunk_mut(direction, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let previous = *value;
                *value = source[start + offset];
                <Self as pcg::Arithmetic<T>>::add_scaled(value, beta, previous);
            }
        });
        true
    }
}

/// Below this, plain sums of products may have lost digits to gradual underflow.
const SAFE_MINIMUM: f64 = f64::MIN_POSITIVE * (1u64 << 52) as f64;

impl<T: Scalar> gmres::Arithmetic<T> for ScalarArithmetic {
    fn zero() -> T {
        T::zero()
    }
    fn to_complex(value: T) -> Complex64 {
        Complex64::new(value.re(), value.im())
    }
    fn from_complex(value: Complex64) -> Option<T> {
        let limit = if T::IS_SINGLE {
            f32::MAX as f64
        } else {
            f64::MAX
        };
        let representable = |component: f64| component.is_finite() && component.abs() <= limit;
        if !representable(value.re)
            || !representable(value.im)
            || (!T::IS_COMPLEX && value.im != 0.0)
        {
            return None;
        }
        Some(T::from_parts(value.re, value.im))
    }
    fn native_norm(values: &[T]) -> Option<f64> {
        // Scaled sum of squares: finite whenever the norm is representable.
        let norm = par::norm2(values);
        norm.is_finite().then_some(norm)
    }
    fn native_dot(left: &[T], right: &[T]) -> Option<Complex64> {
        let (dot, magnitude) = par::dot_and_magnitude(left, right);
        let product = Complex64::new(dot.re(), dot.im());
        (product.re.is_finite()
            && product.im.is_finite()
            && (magnitude == 0.0 || magnitude >= SAFE_MINIMUM))
            .then_some(product)
    }
    fn native_axpy(target: &mut [T], scale: Complex64, source: &[T]) -> bool {
        if (!T::IS_COMPLEX && scale.im != 0.0) || !scale.re.is_finite() || !scale.im.is_finite() {
            return false;
        }
        par::axpy(T::from_parts(scale.re, scale.im), source, target);
        true
    }
    fn native_divide(target: &mut [T], source: &[T], divisor: f64) -> bool {
        if !(divisor.is_finite() && divisor > 0.0) {
            return false;
        }
        par::for_each_chunk_mut(target, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let entry = source[start + offset];
                *value = T::from_parts(entry.re() / divisor, entry.im() / divisor);
            }
        });
        true
    }
    fn divide_complex(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64) {
        // Smith's algorithm: no intermediate |c|² overflow or underflow.
        let ((a, b), (c, d)) = (numerator, denominator);
        if d == 0.0 {
            return (a / c, b / c);
        }
        if c == 0.0 {
            return (b / d, -a / d);
        }
        if c.abs() >= d.abs() {
            let ratio = d / c;
            let scale = c + d * ratio;
            ((a + b * ratio) / scale, (b - a * ratio) / scale)
        } else {
            let ratio = c / d;
            let scale = c * ratio + d;
            ((a * ratio + b) / scale, (b * ratio - a) / scale)
        }
    }
}

/// Recurrence operations over a synchronous operator and preconditioner.
struct SyncOperations<'a, T: Scalar> {
    operator: &'a dyn LinearOperator<T>,
    preconditioner: Option<&'a dyn Preconditioner<T>>,
    control: SolveControl<'a>,
    operator_applications: usize,
    preconditioner_applications: usize,
}

impl<T: Scalar> SyncOperations<'_, T> {
    fn coefficient(&mut self, input: &[T], output: &mut [T]) -> SolverResult<()> {
        self.operator.apply(input, output)?;
        self.operator_applications += 1;
        // Progress counts operator applications; the residual is reported in the final report.
        self.control
            .report_iteration(ProgressPhase::Krylov, self.operator_applications, f64::NAN);
        Ok(())
    }

    fn precondition(&mut self, input: &[T], output: &mut [T]) -> SolverResult<ApplyStatus> {
        let Some(preconditioner) = self.preconditioner else {
            output.copy_from_slice(input);
            return Ok(ApplyStatus::Applied);
        };
        self.preconditioner_applications += 1;
        match preconditioner.apply(input, output) {
            Ok(()) => Ok(ApplyStatus::Applied),
            Err(SolverError::Singular { .. } | SolverError::Breakdown(_)) => {
                Ok(ApplyStatus::NumericalFailure)
            }
            Err(error) => Err(error),
        }
    }
}

impl<T: Scalar> Control for SyncOperations<'_, T> {
    type Error = SolverError;
    fn checkpoint(&mut self) -> SolverResult<()> {
        self.control.check()
    }
    async fn cooperate(&mut self, _completed: u64) -> SolverResult<()> {
        Ok(())
    }
}

impl<T: Scalar> PcgOperations<T> for SyncOperations<'_, T> {
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> SolverResult<ApplyStatus> {
        match operation {
            PcgOperation::Coefficient => self
                .coefficient(input, output)
                .map(|()| ApplyStatus::Applied),
            PcgOperation::FirstPreconditioner | PcgOperation::SecondPreconditioner => {
                self.precondition(input, output)
            }
        }
    }
}

impl<T: Scalar> GmresOperations<T> for SyncOperations<'_, T> {
    type Error = SolverError;
    async fn coefficient(&mut self, input: &[T], output: &mut [T]) -> SolverResult<()> {
        SyncOperations::coefficient(self, input, output)
    }
    async fn precondition(
        &mut self,
        input: &[T],
        output: &mut [T],
        _intermediate: &mut [T],
    ) -> SolverResult<ApplyStatus> {
        SyncOperations::precondition(self, input, output)
    }
    fn checkpoint(&mut self) -> SolverResult<()> {
        self.control.check()
    }
    async fn cooperate(&mut self, _completed: u64) -> SolverResult<()> {
        Ok(())
    }
}

fn failure(error: KrylovFailure<SolverError>) -> SolverError {
    match error {
        KrylovFailure::Operation(original) => original,
        KrylovFailure::Invalid(message) => SolverError::Unsupported(message.to_string()),
    }
}

fn stop_flag(flag: u8) -> StopFlag {
    match flag {
        0 => StopFlag::Converged,
        1 => StopFlag::MaxIterations,
        2 => StopFlag::PreconditionerFailure,
        3 => StopFlag::Stagnation,
        _ => StopFlag::Breakdown,
    }
}

/// Relative tolerance of the recurrence: `rtol`, raised to honor an absolute floor `atol`.
fn relative_tolerance(options: &KrylovOptions, rhs_norm: f64) -> f64 {
    let tolerance = if rhs_norm > 0.0 {
        options.rtol.max(options.atol / rhs_norm)
    } else {
        options.rtol.max(options.atol)
    };
    // The recurrences require a positive tolerance; zero means "iterate to the limit".
    if tolerance.is_finite() && tolerance > 0.0 {
        tolerance
    } else {
        f64::MIN_POSITIVE
    }
}

fn empty_report() -> KrylovReport {
    KrylovReport {
        flag: StopFlag::Converged,
        iterations: 0,
        relative_residual: 0.0,
        residual_norm: 0.0,
        rhs_norm: 0.0,
        residual_history: Vec::new(),
        restart_position: None,
        operator_applications: 0,
        preconditioner_applications: 0,
    }
}

/// `b = 0`: the exact solution of the nonsingular system is `x = 0`, whatever the initial guess
/// (MATLAB returns the same with flag 0 and no iterations).
fn zero_rhs_report<T: Scalar>(x: &mut [T]) -> KrylovReport {
    x.fill(T::zero());
    empty_report()
}

/// PCG workspace entries allocated by [`pcg_solve`] (the caller's `x` and `b` excluded).
pub fn pcg_workspace_len(n: usize, _options: &KrylovOptions) -> usize {
    // best, residual, direction and the operation output vector.
    n * 4
}

/// Blocking PCG for Hermitian positive definite `A` with an HPD preconditioner. `x` holds the
/// initial guess on entry and the solution on return.
pub fn pcg_solve<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let n = b.len();
    if n == 0 {
        return Ok(empty_report());
    }
    if options.max_iterations == 0 {
        return Err(SolverError::Unsupported(
            "PCG needs at least one iteration".into(),
        ));
    }
    PcgGeometry::checked(n, options.max_iterations, false)
        .ok_or_else(|| SolverError::Unsupported("PCG workspace geometry overflows".into()))?;
    let rhs_norm = par::norm2(b);
    if rhs_norm == 0.0 {
        return Ok(zero_rhs_report(x));
    }
    let initial = x
        .iter()
        .any(|value| *value != T::zero())
        .then(|| x.to_vec());
    let mut best = par::try_zeros::<T>(n)?;
    let mut residual = par::try_zeros::<T>(n)?;
    let mut direction = par::try_zeros::<T>(n)?;
    let mut scratch = par::try_zeros::<T>(n)?;
    let mut history = vec![0.0f64; options.max_iterations + 1];
    let mut operations = SyncOperations {
        operator,
        preconditioner,
        control,
        operator_applications: 0,
        preconditioner_applications: 0,
    };
    let outcome = block_on(pcg::solve::<T, ScalarArithmetic, _>(
        b,
        initial.as_deref(),
        PcgParameters {
            tolerance: relative_tolerance(options, rhs_norm),
            maximum_iterations: options.max_iterations,
            preconditioners: (preconditioner.is_some(), false),
        },
        PcgWork {
            x,
            best: &mut best,
            residual: &mut residual,
            direction: &mut direction,
            intermediate: None,
            residual_norms: &mut history,
        },
        &mut scratch,
        &mut operations,
    ))
    .map_err(failure)?;
    history.truncate(outcome.residual_count);
    Ok(KrylovReport {
        flag: stop_flag(outcome.flag),
        iterations: outcome.iterations,
        relative_residual: outcome.relres,
        residual_norm: if rhs_norm > 0.0 {
            outcome.relres * rhs_norm
        } else {
            outcome.relres
        },
        rhs_norm,
        residual_history: if options.record_history {
            history
        } else {
            Vec::new()
        },
        restart_position: None,
        operator_applications: operations.operator_applications,
        preconditioner_applications: operations.preconditioner_applications,
    })
}

/// Options of the synchronous GMRES driver.
#[derive(Debug, Clone, PartialEq)]
pub struct GmresOptions {
    pub restart: usize,
    pub side: PreconditionerSide,
    pub residual_check: ResidualCheck,
}

impl Default for GmresOptions {
    /// Right preconditioning with the true residual formed when the Arnoldi estimate passes.
    fn default() -> Self {
        Self {
            restart: 50,
            side: PreconditionerSide::Right,
            residual_check: ResidualCheck::OnEstimate,
        }
    }
}

/// Workspace entries of the recurrence vectors allocated by [`gmres_solve`].
pub fn gmres_workspace_len(n: usize, _options: &KrylovOptions, gmres: &GmresOptions) -> usize {
    let m = gmres.restart.max(1).min(n.max(1));
    let flexible = if gmres.side == PreconditionerSide::Flexible {
        m
    } else {
        0
    };
    n * (m + 9 + flexible)
}

/// Blocking restarted GMRES / FGMRES for general `A`. `max_iterations` bounds the Arnoldi steps
/// (rounded up to whole restart cycles).
pub fn gmres_solve<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres_options: &GmresOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    check_dims(operator, b, x)?;
    let n = b.len();
    if n == 0 {
        return Ok(empty_report());
    }
    if options.max_iterations == 0 {
        return Err(SolverError::Unsupported(
            "GMRES needs at least one iteration".into(),
        ));
    }
    let restart = gmres_options.restart.max(1).min(n);
    let max_outer = options.max_iterations.div_ceil(restart);
    let geometry = GmresGeometry::checked(n, restart, max_outer)
        .ok_or_else(|| SolverError::Unsupported("GMRES workspace geometry overflows".into()))?;
    let rhs_norm = par::norm2(b);
    if rhs_norm == 0.0 {
        return Ok(zero_rhs_report(x));
    }
    let initial = x
        .iter()
        .any(|value| *value != T::zero())
        .then(|| x.to_vec());
    let mut native = par::try_zeros::<T>(geometry.native_elements)?;
    let mut complex = vec![
        Complex64::new(0.0, 0.0);
        geometry.hessenberg_elements + geometry.complex_small_elements
    ];
    let mut real = vec![0.0f64; geometry.real_small_elements + geometry.history_elements];
    let mode = GmresMode {
        side: gmres_options.side,
        residual_check: gmres_options.residual_check,
        step_limit: (gmres_options.residual_check == ResidualCheck::OnEstimate)
            .then_some(options.max_iterations),
    };
    let mut flexible = if mode.side == PreconditionerSide::Flexible {
        par::try_zeros::<T>(geometry.flexible_elements())?
    } else {
        Vec::new()
    };
    let mut work = geometry
        .project(&mut native, &mut complex, &mut real)
        .ok_or_else(|| SolverError::Unsupported("GMRES workspace projection failed".into()))?;
    let mut operations = SyncOperations {
        operator,
        preconditioner,
        control,
        operator_applications: 0,
        preconditioner_applications: 0,
    };
    let outcome = block_on(gmres::solve_with::<T, ScalarArithmetic, _>(
        b,
        initial.as_deref(),
        GmresParameters {
            restart,
            max_outer,
            tolerance: relative_tolerance(options, rhs_norm),
        },
        mode,
        &mut work,
        &mut flexible,
        &mut operations,
    ))
    .map_err(failure)?;
    x.copy_from_slice(work.x);
    let history = if options.record_history {
        work.history[..outcome.residual_count].to_vec()
    } else {
        Vec::new()
    };
    Ok(KrylovReport {
        flag: stop_flag(outcome.flag),
        iterations: outcome.steps,
        relative_residual: outcome.relres,
        residual_norm: if rhs_norm > 0.0 {
            outcome.relres * rhs_norm
        } else {
            outcome.relres
        },
        rhs_norm,
        residual_history: history,
        restart_position: Some((outcome.outer, outcome.inner)),
        operator_applications: operations.operator_applications,
        preconditioner_applications: operations.preconditioner_applications,
    })
}
