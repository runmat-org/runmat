//! Reliable-residual PCG over exact caller-owned workspace.
use super::{ApplyStatus, Control, KrylovFailure};
mod scale;
mod state;

pub trait Arithmetic<T> {
    const HERMITIAN_EPSILON: f64;
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
    fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> impl std::future::Future<Output = Result<ApplyStatus, Self::Error>>;
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
        parameters.tolerance,
        parameters.maximum_iterations,
        work,
        parameters.preconditioners,
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
