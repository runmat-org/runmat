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
use crate::distributed::Communicator;
use crate::error::{SolverError, SolverResult};
use crate::operator::{LinearOperator, Preconditioner};
use crate::par;
use crate::scalar::Scalar;

/// Zero-sized arithmetic policy of the recurrences for the crate's [`Scalar`] types.
pub struct ScalarArithmetic;

impl<T: Scalar> pcg::Arithmetic<T> for ScalarArithmetic {
    const HERMITIAN_EPSILON: f64 = T::EPSILON;
    const DOMAIN: super::usage::ArithmeticDomain = if T::IS_COMPLEX {
        super::usage::ArithmeticDomain::Complex
    } else {
        super::usage::ArithmeticDomain::Real
    };
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
}

/// Below this, plain sums of products may have lost digits to gradual underflow.
const SAFE_MINIMUM: f64 = f64::MIN_POSITIVE * (1u64 << 52) as f64;

impl<T: Scalar> gmres::Arithmetic<T> for ScalarArithmetic {
    const DOMAIN: super::usage::ArithmeticDomain = if T::IS_COMPLEX {
        super::usage::ArithmeticDomain::Complex
    } else {
        super::usage::ArithmeticDomain::Real
    };
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
    /// Row-distributed solve: reductions gather block partials over this communicator.
    distribution: Option<&'a dyn Communicator>,
    /// A communication failure, reported at the next checkpoint.
    failure: Option<SolverError>,
    capacity_owner: Option<&'a dyn runmat_simulation_usage::AsyncCapacityOwner>,
}

impl<T: Scalar> SyncOperations<'_, T> {
    /// Gathers per-rank encoded partials (rank order); `None` records a failure.
    fn gather(&mut self, comm: &dyn Communicator, local: &[f64]) -> Option<Vec<f64>> {
        match comm.all_gather(local) {
            Ok(all) => Some(all.concat()),
            Err(error) => {
                self.failure.get_or_insert(error);
                None
            }
        }
    }
    /// The same boolean on every rank: all ranks' `local` combined with AND (`any = false`)
    /// or OR (`any = true`).
    fn agree(&mut self, comm: &dyn Communicator, local: bool, any: bool) -> Option<bool> {
        let all = self.gather(comm, &[f64::from(u8::from(local))])?;
        Some(if any {
            all.iter().any(|v| *v != 0.0)
        } else {
            all.iter().all(|v| *v != 0.0)
        })
    }
    /// Global `(Σ conj(l)r, Σ |conj(l)r|)` folded in global block order.
    fn global_dot_magnitude(
        &mut self,
        comm: &dyn Communicator,
        left: &[T],
        right: &[T],
    ) -> Option<(T, f64)> {
        let mut local = Vec::new();
        for (dot, magnitude) in par::dot_and_magnitude_partials(left, right) {
            local.extend([dot.re(), dot.im(), magnitude]);
        }
        let all = self.gather(comm, &local)?;
        Some(par::fold_dot_magnitude_partials(
            all.chunks_exact(3)
                .map(|v| (T::from_parts(v[0], v[1]), v[2])),
        ))
    }
    fn global_norm(&mut self, comm: &dyn Communicator, values: &[T]) -> Option<f64> {
        let local: Vec<f64> = par::norm2_partials(values)
            .into_iter()
            .flat_map(|(s, q)| [s, q])
            .collect();
        let all = self.gather(comm, &local)?;
        Some(par::fold_norm_partials(
            all.chunks_exact(2).map(|v| (v[0], v[1])),
        ))
    }
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
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        self.control.check()
    }
    async fn cooperate(&mut self, _completed: u64) -> SolverResult<()> {
        Ok(())
    }
}

impl<T: Scalar> PcgOperations<T> for SyncOperations<'_, T> {
    async fn measured<R>(
        &mut self,
        work: super::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, super::usage::BatchCompletion),
    ) -> SolverResult<R> {
        let count = work
            .operations()
            .map_err(runmat_simulation_usage::LedgerError::from)?;
        if let Some(owner) = self.capacity_owner {
            owner
                .ensure_capacity(count)
                .await
                .map_err(SolverError::Usage)?;
        }
        let Some(context) = runmat_simulation_usage::UsageContext::capture() else {
            if self.capacity_owner.is_some() {
                return Err(SolverError::Unsupported(
                    "capacity owner requires an active usage context".into(),
                ));
            }
            return Ok(run(self).0);
        };
        let mut reservation = context
            .ledger()
            .reserve(runmat_simulation_usage::UsageCategory::Solve, count)?;
        let (result, completed) = run(self);
        match completed {
            super::usage::BatchCompletion::Skipped => {}
            super::usage::BatchCompletion::Complete => reservation.complete(count)?,
            super::usage::BatchCompletion::Prefix(prefix) => reservation.complete(prefix)?,
        }
        Ok(result)
    }

    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> SolverResult<ApplyStatus> {
        if let Some(owner) = self.capacity_owner {
            let work = match operation {
                PcgOperation::Coefficient => self.operator.reference_apply_work()?,
                PcgOperation::FirstPreconditioner | PcgOperation::SecondPreconditioner => {
                    match self.preconditioner {
                        Some(preconditioner) => preconditioner.reference_apply_work()?,
                        None => None,
                    }
                }
            }
            .ok_or_else(|| {
                SolverError::Unsupported(
                    "awaited PCG application has no exact reference-work quote".into(),
                )
            })?;
            owner
                .ensure_capacity(work)
                .await
                .map_err(SolverError::Usage)?;
        }
        match operation {
            PcgOperation::Coefficient => self
                .coefficient(input, output)
                .map(|()| ApplyStatus::Applied),
            PcgOperation::FirstPreconditioner | PcgOperation::SecondPreconditioner => {
                self.precondition(input, output)
            }
        }
    }
    fn native_dot(&mut self, left: &[T], right: &[T]) -> Option<(f64, f64, f64)> {
        let Some(comm) = self.distribution else {
            return HostKernels::pcg_dot(left, right);
        };
        let (dot, magnitude) = self.global_dot_magnitude(comm, left, right)?;
        HostKernels::accept_pcg_dot(dot, magnitude)
    }
    fn native_step(
        &mut self,
        x: &mut [T],
        residual: &mut [T],
        direction: &[T],
        product: &[T],
        alpha: f64,
    ) -> Option<bool> {
        let progressed = HostKernels::pcg_step(x, residual, direction, product, alpha)?;
        match self.distribution {
            Some(comm) => self.agree(comm, progressed, true),
            None => Some(progressed),
        }
    }
    fn native_direction(&mut self, direction: &mut [T], source: &[T], beta: f64) -> bool {
        HostKernels::pcg_direction(direction, source, beta)
    }
    fn native_residual(&mut self, rhs: &[T], product: &[T], residual: &mut [T]) -> bool {
        HostKernels::pcg_residual(rhs, product, residual)
    }
}

impl<T: Scalar> GmresOperations<T> for SyncOperations<'_, T> {
    type Error = SolverError;
    async fn measured<R>(
        &mut self,
        work: super::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, super::usage::BatchCompletion),
    ) -> SolverResult<R> {
        <Self as PcgOperations<T>>::measured(self, work, run).await
    }
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
        <Self as Control>::checkpoint(self)
    }
    async fn cooperate(&mut self, _completed: u64) -> SolverResult<()> {
        Ok(())
    }
    fn native_norm(&mut self, values: &[T]) -> Option<f64> {
        let Some(comm) = self.distribution else {
            return HostKernels::norm(values);
        };
        self.global_norm(comm, values)
            .filter(|norm| norm.is_finite())
    }
    fn native_dot(&mut self, left: &[T], right: &[T]) -> Option<Complex64> {
        let Some(comm) = self.distribution else {
            return HostKernels::dot(left, right);
        };
        let (dot, magnitude) = self.global_dot_magnitude(comm, left, right)?;
        HostKernels::accept_dot(dot, magnitude)
    }
    fn native_axpy(&mut self, target: &mut [T], scale: Complex64, source: &[T]) -> bool {
        HostKernels::axpy(target, scale, source)
    }
    fn native_project(
        &mut self,
        basis: &[T],
        rows: usize,
        w: &mut [T],
        coefficients: &mut [Complex64],
    ) -> bool {
        let Some(comm) = self.distribution else {
            return HostKernels::project(basis, rows, w, coefficients);
        };
        let n = w.len();
        if basis.len() != rows * n || coefficients.len() != rows {
            return false;
        }
        let mut local = Vec::new();
        for block in par::multi_dot_partials(basis, rows, w) {
            for value in block {
                local.extend([value.re(), value.im()]);
            }
        }
        let Some(all) = self.gather(comm, &local) else {
            return false;
        };
        let dots = par::fold_multi_partials(
            rows,
            all.chunks_exact(2 * rows).map(|block| {
                block
                    .chunks_exact(2)
                    .map(|v| T::from_parts(v[0], v[1]))
                    .collect()
            }),
        );
        HostKernels::apply_projection(basis, &dots, w, coefficients)
    }
    fn native_divide(&mut self, target: &mut [T], source: &[T], divisor: f64) -> bool {
        HostKernels::divide(target, source, divisor)
    }
    fn native_residual(&mut self, rhs: &[T], product: &[T], residual: &mut [T]) -> Option<bool> {
        let finite = HostKernels::residual(rhs, product, residual)?;
        match self.distribution {
            Some(comm) => self.agree(comm, finite, false),
            None => Some(finite),
        }
    }
    fn native_equal(&mut self, left: &[T], right: &[T]) -> Option<bool> {
        let equal = HostKernels::equal(left, right)?;
        match self.distribution {
            Some(comm) => self.agree(comm, equal, false),
            None => Some(equal),
        }
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
        spectrum: None,
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
    pcg_impl(operator, preconditioner, b, x, options, control, None)
}

/// PCG over the same recurrence with awaited capacity for each measured batch.
pub async fn pcg_solve_with_capacity<T: Scalar, E>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<KrylovReport, PcgCapacityError<E>> {
    pcg_impl_typed(operator, preconditioner, b, x, options, control, owner).await
}

pub enum PcgCapacityError<E> {
    Numerical(SolverError),
    Admission(E),
}

impl<E> From<SolverError> for PcgCapacityError<E> {
    fn from(error: SolverError) -> Self {
        Self::Numerical(error)
    }
}

struct OwnedPcgOperations<'a, T: Scalar, E> {
    inner: SyncOperations<'a, T>,
    owner: &'a dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
}

impl<T: Scalar, E> Control for OwnedPcgOperations<'_, T, E> {
    type Error = PcgCapacityError<E>;
    fn checkpoint(&mut self) -> Result<(), Self::Error> {
        <SyncOperations<'_, T> as Control>::checkpoint(&mut self.inner).map_err(Into::into)
    }
    async fn cooperate(&mut self, completed: u64) -> Result<(), Self::Error> {
        <SyncOperations<'_, T> as Control>::cooperate(&mut self.inner, completed)
            .await
            .map_err(Into::into)
    }
}

impl<T: Scalar, E> PcgOperations<T> for OwnedPcgOperations<'_, T, E> {
    async fn measured<R>(
        &mut self,
        work: super::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, super::usage::BatchCompletion),
    ) -> Result<R, Self::Error> {
        let count = work
            .operations()
            .map_err(runmat_simulation_usage::LedgerError::from)
            .map_err(SolverError::Usage)?;
        self.owner
            .ensure_capacity(count)
            .await
            .map_err(PcgCapacityError::Admission)?;
        let context = runmat_simulation_usage::UsageContext::capture().ok_or_else(|| {
            SolverError::Unsupported("capacity owner requires an active usage context".into())
        })?;
        let mut reservation = context
            .ledger()
            .reserve(runmat_simulation_usage::UsageCategory::Solve, count)
            .map_err(SolverError::Usage)?;
        let (result, completed) = run(self);
        match completed {
            super::usage::BatchCompletion::Skipped => {}
            super::usage::BatchCompletion::Complete => {
                reservation.complete(count).map_err(SolverError::Usage)?
            }
            super::usage::BatchCompletion::Prefix(prefix) => {
                reservation.complete(prefix).map_err(SolverError::Usage)?
            }
        }
        Ok(result)
    }

    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> Result<ApplyStatus, Self::Error> {
        let work = match operation {
            PcgOperation::Coefficient => self.inner.operator.reference_apply_work()?,
            PcgOperation::FirstPreconditioner | PcgOperation::SecondPreconditioner => {
                match self.inner.preconditioner {
                    Some(preconditioner) => preconditioner.reference_apply_work()?,
                    None => None,
                }
            }
        }
        .ok_or_else(|| {
            SolverError::Unsupported(
                "awaited PCG application has no exact reference-work quote".into(),
            )
        })?;
        self.owner
            .ensure_capacity(work)
            .await
            .map_err(PcgCapacityError::Admission)?;
        <SyncOperations<'_, T> as PcgOperations<T>>::apply(
            &mut self.inner,
            operation,
            input,
            output,
        )
        .await
        .map_err(Into::into)
    }

    fn native_dot(&mut self, left: &[T], right: &[T]) -> Option<(f64, f64, f64)> {
        <SyncOperations<'_, T> as PcgOperations<T>>::native_dot(&mut self.inner, left, right)
    }
    fn native_step(
        &mut self,
        x: &mut [T],
        residual: &mut [T],
        direction: &[T],
        product: &[T],
        alpha: f64,
    ) -> Option<bool> {
        <SyncOperations<'_, T> as PcgOperations<T>>::native_step(
            &mut self.inner,
            x,
            residual,
            direction,
            product,
            alpha,
        )
    }
    fn native_direction(&mut self, direction: &mut [T], source: &[T], beta: f64) -> bool {
        <SyncOperations<'_, T> as PcgOperations<T>>::native_direction(
            &mut self.inner,
            direction,
            source,
            beta,
        )
    }
    fn native_residual(&mut self, rhs: &[T], product: &[T], residual: &mut [T]) -> bool {
        <SyncOperations<'_, T> as PcgOperations<T>>::native_residual(
            &mut self.inner,
            rhs,
            product,
            residual,
        )
    }
}

async fn pcg_impl_typed<T: Scalar, E>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<KrylovReport, PcgCapacityError<E>> {
    check_dims(operator, b, x)?;
    let n = b.len();
    if n == 0 {
        return Ok(empty_report());
    }
    if options.max_iterations == 0 {
        return Err(SolverError::Unsupported("PCG needs at least one iteration".into()).into());
    }
    PcgGeometry::checked(n, options.max_iterations, false)
        .ok_or_else(|| SolverError::Unsupported("PCG workspace geometry overflows".into()))?;
    let norm_work = runmat_simulation_usage::OperationCount::from_operations(
        u64::try_from(n)
            .map_err(|_| SolverError::Usage(runmat_simulation_usage::LedgerError::CountOverflow))?,
    )
    .checked_mul(if T::IS_COMPLEX { 4 } else { 2 })
    .map_err(runmat_simulation_usage::LedgerError::from)
    .map_err(SolverError::Usage)?;
    owner
        .ensure_capacity(norm_work)
        .await
        .map_err(PcgCapacityError::Admission)?;
    let context = runmat_simulation_usage::UsageContext::capture().ok_or_else(|| {
        SolverError::Unsupported("capacity owner requires an active usage context".into())
    })?;
    let mut reservation = context
        .ledger()
        .reserve(runmat_simulation_usage::UsageCategory::Solve, norm_work)
        .map_err(SolverError::Usage)?;
    let rhs_norm = rhs_norm(b, None)?;
    reservation
        .complete(norm_work)
        .map_err(SolverError::Usage)?;
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
    let flexible = preconditioner.is_some_and(|p| p.is_variable());
    let mut previous = if flexible {
        par::try_zeros::<T>(n)?
    } else {
        Vec::new()
    };
    let mut coefficients = super::pcg::PcgCoefficients::default();
    let inner = SyncOperations {
        operator,
        preconditioner,
        control,
        operator_applications: 0,
        preconditioner_applications: 0,
        distribution: None,
        failure: None,
        capacity_owner: None,
    };
    let mut operations = OwnedPcgOperations { inner, owner };
    let outcome = pcg::solve_with::<T, ScalarArithmetic, _>(
        b,
        initial.as_deref(),
        PcgParameters {
            tolerance: relative_tolerance(options, rhs_norm),
            maximum_iterations: options.max_iterations,
            preconditioners: (preconditioner.is_some(), false),
        },
        pcg::PcgMode { flexible },
        PcgWork {
            x,
            best: &mut best,
            residual: &mut residual,
            direction: &mut direction,
            intermediate: None,
            residual_norms: &mut history,
            coefficients: Some(&mut coefficients),
        },
        &mut previous,
        &mut scratch,
        &mut operations,
    )
    .await
    .map_err(|failure| match failure {
        KrylovFailure::Operation(error) => error,
        KrylovFailure::Invalid(message) => {
            PcgCapacityError::Numerical(SolverError::Unsupported(message.into()))
        }
    })?;
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
        operator_applications: operations.inner.operator_applications,
        preconditioner_applications: operations.inner.preconditioner_applications,
        spectrum: super::lanczos::PcgSpectrumEstimate::from_coefficients(&coefficients),
    })
}

/// Row-distributed PCG: `operator`, `b` and `x` are this rank's owned rows (the operator does
/// its own halo exchange, e.g. [`crate::distributed::DistributedCsr`]), and every rank calls
/// collectively. With block-aligned rank boundaries the iterates are bitwise those of the
/// single-rank solve.
pub fn pcg_distributed<T: Scalar>(
    comm: &dyn Communicator,
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    pcg_impl(operator, preconditioner, b, x, options, control, Some(comm))
}

/// `‖b‖`, globally reduced for a distributed solve.
fn rhs_norm<T: Scalar>(b: &[T], distribution: Option<&dyn Communicator>) -> SolverResult<f64> {
    let Some(comm) = distribution else {
        return Ok(par::norm2(b));
    };
    let local: Vec<f64> = par::norm2_partials(b)
        .into_iter()
        .flat_map(|(s, q)| [s, q])
        .collect();
    let all = comm.all_gather(&local)?.concat();
    Ok(par::fold_norm_partials(
        all.chunks_exact(2).map(|v| (v[0], v[1])),
    ))
}

#[allow(clippy::too_many_arguments)]
fn pcg_impl<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
    distribution: Option<&dyn Communicator>,
) -> SolverResult<KrylovReport> {
    block_on(pcg_impl_async(
        operator,
        preconditioner,
        b,
        x,
        options,
        control,
        distribution,
    ))
}

async fn pcg_impl_async<T: Scalar>(
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    control: SolveControl<'_>,
    distribution: Option<&dyn Communicator>,
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
    let norm_work = runmat_simulation_usage::OperationCount::from_operations(
        u64::try_from(n).map_err(|_| runmat_simulation_usage::LedgerError::CountOverflow)?,
    )
    .checked_mul(if T::IS_COMPLEX { 4 } else { 2 })
    .map_err(runmat_simulation_usage::LedgerError::from)?;
    let rhs_norm = match runmat_simulation_usage::UsageContext::capture() {
        Some(context) => {
            let mut reservation = context
                .ledger()
                .reserve(runmat_simulation_usage::UsageCategory::Solve, norm_work)?;
            let value = rhs_norm(b, distribution)?;
            reservation.complete(norm_work)?;
            value
        }
        None => rhs_norm(b, distribution)?,
    };
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
    // A variable preconditioner needs flexible CG (one extra vector).
    let flexible = preconditioner.is_some_and(|p| p.is_variable());
    let mut previous = if flexible {
        par::try_zeros::<T>(n)?
    } else {
        Vec::new()
    };
    let mut coefficients = super::pcg::PcgCoefficients::default();
    let mut operations = SyncOperations {
        operator,
        preconditioner,
        control,
        operator_applications: 0,
        preconditioner_applications: 0,
        distribution,
        failure: None,
        capacity_owner: None,
    };
    let outcome = pcg::solve_with::<T, ScalarArithmetic, _>(
        b,
        initial.as_deref(),
        PcgParameters {
            tolerance: relative_tolerance(options, rhs_norm),
            maximum_iterations: options.max_iterations,
            preconditioners: (preconditioner.is_some(), false),
        },
        pcg::PcgMode { flexible },
        PcgWork {
            x,
            best: &mut best,
            residual: &mut residual,
            direction: &mut direction,
            intermediate: None,
            residual_norms: &mut history,
            coefficients: Some(&mut coefficients),
        },
        &mut previous,
        &mut scratch,
        &mut operations,
    )
    .await
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
        spectrum: super::lanczos::PcgSpectrumEstimate::from_coefficients(&coefficients),
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
    gmres_with_workspace(
        &mut GmresWorkspace::default(),
        operator,
        preconditioner,
        b,
        x,
        options,
        gmres_options,
        control,
    )
}

/// Caller-owned GMRES storage reused across solves (inner solves of a preconditioner, sweeps).
/// Every region is written by the recurrence before it is read, so reuse needs no clearing.
#[derive(Debug, Default)]
pub struct GmresWorkspace<T> {
    native: Vec<T>,
    complex: Vec<Complex64>,
    real: Vec<f64>,
    flexible: Vec<T>,
}

impl<T: Scalar> GmresWorkspace<T> {
    /// Bytes currently held.
    pub fn retained_bytes(&self) -> u64 {
        ((self.native.capacity() + self.flexible.capacity()) * std::mem::size_of::<T>()
            + self.complex.capacity() * 16
            + self.real.capacity() * 8) as u64
    }
}

fn exact_len<V: Clone>(vector: &mut Vec<V>, len: usize, zero: V) -> SolverResult<()> {
    if vector.len() < len {
        vector
            .try_reserve_exact(len - vector.len())
            .map_err(|_| SolverError::OutOfMemory {
                bytes: (len * std::mem::size_of::<V>()) as u64,
            })?;
    }
    vector.resize(len, zero);
    Ok(())
}

/// [`gmres_solve`] on caller-owned, reusable storage.
#[allow(clippy::too_many_arguments)]
pub fn gmres_with_workspace<T: Scalar>(
    workspace: &mut GmresWorkspace<T>,
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres_options: &GmresOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    gmres_impl(
        workspace,
        operator,
        preconditioner,
        b,
        x,
        options,
        gmres_options,
        control,
        None,
    )
}

/// Row-distributed GMRES (see [`pcg_distributed`]).
#[allow(clippy::too_many_arguments)]
pub fn gmres_distributed<T: Scalar>(
    comm: &dyn Communicator,
    workspace: &mut GmresWorkspace<T>,
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres_options: &GmresOptions,
    control: SolveControl<'_>,
) -> SolverResult<KrylovReport> {
    gmres_impl(
        workspace,
        operator,
        preconditioner,
        b,
        x,
        options,
        gmres_options,
        control,
        Some(comm),
    )
}

#[allow(clippy::too_many_arguments)]
fn gmres_impl<T: Scalar>(
    workspace: &mut GmresWorkspace<T>,
    operator: &dyn LinearOperator<T>,
    preconditioner: Option<&dyn Preconditioner<T>>,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres_options: &GmresOptions,
    control: SolveControl<'_>,
    distribution: Option<&dyn Communicator>,
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
    let rhs_norm = rhs_norm(b, distribution)?;
    if rhs_norm == 0.0 {
        return Ok(zero_rhs_report(x));
    }
    let initial = x
        .iter()
        .any(|value| *value != T::zero())
        .then(|| x.to_vec());
    let GmresWorkspace {
        native,
        complex,
        real,
        flexible,
    } = workspace;
    exact_len(native, geometry.native_elements, T::zero())?;
    exact_len(
        complex,
        geometry.hessenberg_elements + geometry.complex_small_elements,
        Complex64::new(0.0, 0.0),
    )?;
    exact_len(
        real,
        geometry.real_small_elements + geometry.history_elements,
        0.0,
    )?;
    let mode = GmresMode {
        // A variable preconditioner needs the flexible variant.
        side: if preconditioner.is_some_and(|p| p.is_variable()) {
            PreconditionerSide::Flexible
        } else {
            gmres_options.side
        },
        residual_check: gmres_options.residual_check,
        step_limit: (gmres_options.residual_check == ResidualCheck::OnEstimate)
            .then_some(options.max_iterations),
    };
    let flexible_len = if mode.side == PreconditionerSide::Flexible {
        geometry.flexible_elements()
    } else {
        0
    };
    exact_len(flexible, flexible_len, T::zero())?;
    let mut work = geometry
        .project(native, complex, real)
        .ok_or_else(|| SolverError::Unsupported("GMRES workspace projection failed".into()))?;
    let mut operations = SyncOperations {
        operator,
        preconditioner,
        control,
        operator_applications: 0,
        preconditioner_applications: 0,
        distribution,
        failure: None,
        capacity_owner: None,
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
        flexible,
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
        spectrum: None,
    })
}

/// The host (shared-memory) vector kernels of the recurrences: deterministic blocked `par`
/// kernels on the admitted pool. Operation objects supply them through the native hooks of
/// [`GmresOperations`] and [`PcgOperations`]; [`SyncOperations`] and FEA's closure-operator
/// GMRES delegate here.
pub struct HostKernels;

impl HostKernels {
    pub fn pcg_dot<T: Scalar>(left: &[T], right: &[T]) -> Option<(f64, f64, f64)> {
        let (dot, magnitude) = par::dot_and_magnitude(left, right);
        Self::accept_pcg_dot(dot, magnitude)
    }

    /// Plain sums are exact enough when finite and clear of gradual underflow.
    pub fn accept_pcg_dot<T: Scalar>(dot: T, magnitude: f64) -> Option<(f64, f64, f64)> {
        (dot.re().is_finite()
            && dot.im().is_finite()
            && magnitude.is_finite()
            && (magnitude == 0.0 || magnitude >= SAFE_MINIMUM))
            .then(|| (dot.re(), dot.im(), magnitude))
    }

    pub fn pcg_step<T: Scalar>(
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
                <ScalarArithmetic as pcg::Arithmetic<T>>::add_scaled(
                    value,
                    alpha,
                    direction[start + offset],
                );
                changed |= *value != previous;
            }
            if changed {
                progressed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        par::for_each_chunk_mut(residual, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                <ScalarArithmetic as pcg::Arithmetic<T>>::subtract_scaled(
                    value,
                    alpha,
                    product[start + offset],
                );
            }
        });
        Some(progressed.into_inner())
    }

    pub fn pcg_direction<T: Scalar>(direction: &mut [T], source: &[T], beta: f64) -> bool {
        par::for_each_chunk_mut(direction, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let previous = *value;
                *value = source[start + offset];
                <ScalarArithmetic as pcg::Arithmetic<T>>::add_scaled(value, beta, previous);
            }
        });
        true
    }

    pub fn pcg_residual<T: Scalar>(rhs: &[T], product: &[T], residual: &mut [T]) -> bool {
        par::sub_into(rhs, product, residual);
        true
    }

    pub fn norm<T: Scalar>(values: &[T]) -> Option<f64> {
        // Scaled sum of squares: finite whenever the norm is representable.
        let norm = par::norm2(values);
        norm.is_finite().then_some(norm)
    }

    pub fn dot<T: Scalar>(left: &[T], right: &[T]) -> Option<Complex64> {
        let (dot, magnitude) = par::dot_and_magnitude(left, right);
        Self::accept_dot(dot, magnitude)
    }

    pub fn accept_dot<T: Scalar>(dot: T, magnitude: f64) -> Option<Complex64> {
        let product = Complex64::new(dot.re(), dot.im());
        (product.re.is_finite()
            && product.im.is_finite()
            && (magnitude == 0.0 || magnitude >= SAFE_MINIMUM))
            .then_some(product)
    }

    pub fn axpy<T: Scalar>(target: &mut [T], scale: Complex64, source: &[T]) -> bool {
        if (!T::IS_COMPLEX && scale.im != 0.0) || !scale.re.is_finite() || !scale.im.is_finite() {
            return false;
        }
        par::axpy(T::from_parts(scale.re, scale.im), source, target);
        true
    }

    pub fn project<T: Scalar>(
        basis: &[T],
        rows: usize,
        w: &mut [T],
        coefficients: &mut [Complex64],
    ) -> bool {
        let n = w.len();
        if basis.len() != rows * n || coefficients.len() != rows {
            return false;
        }
        let dots = par::multi_dot(basis, rows, w);
        Self::apply_projection(basis, &dots, w, coefficients)
    }

    /// `w −= Σ dots[k] v_k` after checking the (globally reduced) coefficients.
    pub fn apply_projection<T: Scalar>(
        basis: &[T],
        dots: &[T],
        w: &mut [T],
        coefficients: &mut [Complex64],
    ) -> bool {
        if dots.iter().any(|value| !value.is_finite()) {
            return false;
        }
        for (target, value) in coefficients.iter_mut().zip(dots) {
            *target = Complex64::new(value.re(), value.im());
        }
        par::multi_axpy(basis, dots, w, -1.0);
        true
    }

    pub fn divide<T: Scalar>(target: &mut [T], source: &[T], divisor: f64) -> bool {
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

    pub fn residual<T: Scalar>(rhs: &[T], product: &[T], residual: &mut [T]) -> Option<bool> {
        par::sub_into(rhs, product, residual);
        Some(residual.iter().all(|value| value.is_finite()))
    }

    pub fn equal<T: Scalar>(left: &[T], right: &[T]) -> Option<bool> {
        Some(left == right)
    }
}
