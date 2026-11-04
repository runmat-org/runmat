//! Reliable-residual PCG over exact caller-owned workspace.
use super::{ApplyStatus, Control, KrylovFailure};
mod scale;
mod state;

pub trait Arithmetic<T> {
    const HERMITIAN_EPSILON: f64;
    const DOMAIN: super::usage::ArithmeticDomain;
    fn zero() -> T;
    fn components(value: T) -> (f64, f64);
    fn add_scaled(target: &mut T, scale: f64, value: T);
    fn subtract_scaled(target: &mut T, scale: f64, value: T);
    fn scaled_conjugate_product(left: T, right: T, left_scale: f64, right_scale: f64)
        -> (f64, f64);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcgOperation {
    Coefficient,
    FirstPreconditioner,
    SecondPreconditioner,
}
enum PcgContinuation<'a, T> {
    Apply {
        operation: PcgOperation,
        input: &'a [T],
    },
    Complete(PcgOutcome),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgOutcome {
    pub flag: u8,
    pub relres: f64,
    pub iterations: usize,
    pub residual_count: usize,
}
pub struct PcgWork<'a, T> {
    pub x: &'a mut [T],
    pub best: &'a mut [T],
    pub residual: &'a mut [T],
    pub direction: &'a mut [T],
    pub intermediate: Option<&'a mut [T]>,
    pub residual_norms: &'a mut [f64],
    /// Receives the CG coefficients of the final Lanczos segment when present.
    pub coefficients: Option<&'a mut PcgCoefficients>,
}

/// The CG coefficients of the current Lanczos segment, from the start or the last reliable
/// residual restart: `α_j = ρ_j / ⟨p_j, A p_j⟩` and `β_j = ρ_{j+1} / ρ_j`. They define the
/// Lanczos tridiagonal of the preconditioned operator `M⁻¹A` (see [`super::lanczos`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PcgCoefficients {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    /// Flexible mode: β is Polak–Ribière, so the tridiagonal is only approximately Lanczos.
    pub flexible: bool,
}

impl PcgCoefficients {
    fn restart(&mut self) {
        self.alpha.clear();
        self.beta.clear();
    }
}
#[derive(Clone, Copy)]
pub struct PcgParameters {
    pub tolerance: f64,
    pub maximum_iterations: usize,
    pub preconditioners: (bool, bool),
}

/// Includes the separately borrowed operation-output vector and all history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcgGeometry {
    pub native_elements: usize,
    pub history_elements: usize,
}
impl PcgGeometry {
    pub fn checked(
        n: usize,
        maximum_iterations: usize,
        both_preconditioners: bool,
    ) -> Option<Self> {
        if n == 0 || maximum_iterations == 0 {
            return None;
        }
        Some(Self {
            native_elements: n.checked_mul(if both_preconditioners { 6 } else { 5 })?,
            history_elements: maximum_iterations.checked_add(1)?,
        })
    }
}

pub trait PcgOperations<T>: Control {
    /// Admit a typed batch before running it and complete it only after the
    /// closure returns. Offline callers explicitly run the closure unchanged.
    fn measured<R>(
        &mut self,
        work: super::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, super::usage::BatchCompletion),
    ) -> impl std::future::Future<Output = Result<R, Self::Error>>;

    fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> impl std::future::Future<Output = Result<ApplyStatus, Self::Error>>;

    // Optional native vector kernels (see `gmres::GmresOperations`): `None`/`false` (the
    // default) selects the portable kernels; native answers are identical on every rank.

    /// `(Re Σ conj(l)r, Im Σ conj(l)r, Σ |conj(l)r|)`; `None` also when a native sum is not
    /// finite and safely above underflow.
    fn native_dot(&mut self, _left: &[T], _right: &[T]) -> Option<(f64, f64, f64)> {
        None
    }
    /// `x += α p; r −= α q`, returning whether any entry of `x` changed.
    fn native_step(
        &mut self,
        _x: &mut [T],
        _residual: &mut [T],
        _direction: &[T],
        _product: &[T],
        _alpha: f64,
    ) -> Option<bool> {
        None
    }
    /// `p = s + β p`; `false` leaves `p` untouched.
    fn native_direction(&mut self, _direction: &mut [T], _source: &[T], _beta: f64) -> bool {
        false
    }
    /// `residual = rhs − product`; `false` selects the portable loop.
    fn native_residual(&mut self, _rhs: &[T], _product: &[T], _residual: &mut [T]) -> bool {
        false
    }
}

/// Iteration variant of the one PCG recurrence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PcgMode {
    /// Flexible CG (Polak–Ribière β, FCG(1)): valid when the preconditioner varies between
    /// applications (an inner iterative solve or a reduced-precision cycle). Needs one extra
    /// vector (`flexible_previous`) holding the previous residual.
    pub flexible: bool,
}

/// The state is consumed on error. Each emitted A/M1/M2 request is applied
/// exactly once; convergence and returned-best certificates are never replayed.
pub async fn solve<'a, T: Copy + PartialEq, A: Arithmetic<T>, O: PcgOperations<T>>(
    rhs: &'a [T],
    initial: Option<&[T]>,
    parameters: PcgParameters,
    work: PcgWork<'a, T>,
    operation_scratch: &mut [T],
    operations: &mut O,
) -> Result<PcgOutcome, KrylovFailure<O::Error>> {
    solve_with::<T, A, O>(
        rhs,
        initial,
        parameters,
        PcgMode::default(),
        work,
        &mut [],
        operation_scratch,
        operations,
    )
    .await
}

/// PCG in the selected [`PcgMode`]; `flexible_previous` has `rhs.len()` entries when flexible
/// and may be empty otherwise.
#[allow(clippy::too_many_arguments)]
pub async fn solve_with<'a, T: Copy + PartialEq, A: Arithmetic<T>, O: PcgOperations<T>>(
    rhs: &'a [T],
    initial: Option<&[T]>,
    parameters: PcgParameters,
    mode: PcgMode,
    work: PcgWork<'a, T>,
    flexible_previous: &'a mut [T],
    operation_scratch: &mut [T],
    operations: &mut O,
) -> Result<PcgOutcome, KrylovFailure<O::Error>> {
    if mode.flexible && flexible_previous.len() != rhs.len() {
        return Err(KrylovFailure::Invalid(
            "flexible PCG previous-residual extent differs from b",
        ));
    }
    PcgGeometry::checked(
        rhs.len(),
        parameters.maximum_iterations,
        parameters.preconditioners.0 && parameters.preconditioners.1,
    )
    .ok_or(KrylovFailure::Invalid(
        "PCG workspace geometry overflows or is empty",
    ))?;
    if operation_scratch.len() != rhs.len()
        || !parameters.tolerance.is_finite()
        || parameters.tolerance <= 0.0
    {
        return Err(KrylovFailure::Invalid(
            "PCG parameters or operation extent are invalid",
        ));
    }
    let mut state = state::RecurrenceState::<T, A>::new(
        rhs,
        initial,
        parameters,
        work,
        mode.flexible.then_some(flexible_previous),
        operations,
    )
    .await?;
    loop {
        operations.checkpoint().map_err(KrylovFailure::Operation)?;
        let (operation, input) = match state.continuation() {
            PcgContinuation::Complete(outcome) => return Ok(outcome),
            PcgContinuation::Apply { operation, input } => (operation, input),
        };
        let status = operations
            .apply(operation, input, operation_scratch)
            .await
            .map_err(KrylovFailure::Operation)?;
        operations.checkpoint().map_err(KrylovFailure::Operation)?;
        match status {
            ApplyStatus::Applied => state.accept(operation_scratch, operations).await?,
            ApplyStatus::NumericalFailure => {
                if operation == PcgOperation::Coefficient {
                    return Err(KrylovFailure::Invalid(
                        "coefficient returned a preconditioner failure",
                    ));
                }
                state.preconditioner_failed(operations).await?;
            }
        }
    }
}
