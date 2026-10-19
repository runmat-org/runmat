//! Native-precision PCG scalars and the synchronous driver of the one
//! recurrence shared with the asynchronous callable route.

use runmat_value::ComplexElement;

use super::async_driver::PcgOperation;
use super::recurrence_state::{drive_sync, RecurrenceState};
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::RuntimeContext;
use crate::RuntimeError;

pub(super) trait PcgScalar: Copy + PartialEq {
    const HERMITIAN_EPSILON: f64;
    fn zero() -> Self;
    fn add_scaled(&mut self, scale: f64, value: Self);
    fn subtract_scaled(&mut self, scale: f64, value: Self);
    fn magnitude(self) -> f64;
    fn scaled_conjugate_product(self, other: Self, left: f64, right: f64) -> (f64, f64);
}

macro_rules! real_scalar {
    ($scalar:ty) => {
        impl PcgScalar for $scalar {
            const HERMITIAN_EPSILON: f64 = <$scalar>::EPSILON as f64;
            fn zero() -> Self {
                0.0
            }
            fn add_scaled(&mut self, scale: f64, value: Self) {
                *self = (f64::from(*self) + scale * f64::from(value)) as $scalar;
            }
            fn subtract_scaled(&mut self, scale: f64, value: Self) {
                *self = (f64::from(*self) - scale * f64::from(value)) as $scalar;
            }
            fn magnitude(self) -> f64 {
                f64::from(self).abs()
            }
            fn scaled_conjugate_product(self, other: Self, left: f64, right: f64) -> (f64, f64) {
                ((f64::from(self) / left) * (f64::from(other) / right), 0.0)
            }
        }
    };
}

macro_rules! complex_scalar {
    ($scalar:ty) => {
        impl PcgScalar for ComplexElement<$scalar> {
            const HERMITIAN_EPSILON: f64 = <$scalar>::EPSILON as f64;
            fn zero() -> Self {
                ComplexElement(0.0, 0.0)
            }
            fn add_scaled(&mut self, scale: f64, value: Self) {
                self.0 = (f64::from(self.0) + scale * f64::from(value.0)) as $scalar;
                self.1 = (f64::from(self.1) + scale * f64::from(value.1)) as $scalar;
            }
            fn subtract_scaled(&mut self, scale: f64, value: Self) {
                self.0 = (f64::from(self.0) - scale * f64::from(value.0)) as $scalar;
                self.1 = (f64::from(self.1) - scale * f64::from(value.1)) as $scalar;
            }
            fn magnitude(self) -> f64 {
                f64::from(self.0).hypot(f64::from(self.1))
            }
            fn scaled_conjugate_product(self, other: Self, left: f64, right: f64) -> (f64, f64) {
                let ar = f64::from(self.0) / left;
                let ai = f64::from(self.1) / left;
                let br = f64::from(other.0) / right;
                let bi = f64::from(other.1) / right;
                (ar * br + ai * bi, ar * bi - ai * br)
            }
        }
    };
}

real_scalar!(f32);
real_scalar!(f64);
complex_scalar!(f32);
complex_scalar!(f64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PcgOutcome {
    pub(super) flag: u8,
    pub(super) relres: f64,
    pub(super) iterations: usize,
    pub(super) residual_count: usize,
}

/// Four always-live vectors plus an intermediate only when M1 feeds M2. The
/// caller owns a fifth operation-output vector independently of this state.
pub(super) struct PcgWork<'a, T> {
    pub(super) x: &'a mut [T],
    pub(super) best: &'a mut [T],
    pub(super) residual: &'a mut [T],
    pub(super) direction: &'a mut [T],
    pub(super) intermediate: Option<&'a mut [T]>,
    pub(super) residual_norms: &'a mut [f64],
}

pub(super) struct PcgSolveInput<'a, T: PcgScalar> {
    pub(super) rhs: &'a [T],
    pub(super) initial: Option<&'a [T]>,
    pub(super) tolerance: f64,
    pub(super) maximum_iterations: usize,
    pub(super) work: PcgWork<'a, T>,
    pub(super) operation_scratch: &'a mut [T],
}

pub(super) fn solve<T: PcgScalar>(
    input: PcgSolveInput<'_, T>,
    context: Option<&RuntimeContext>,
    mut matvec: impl FnMut(&[T], &mut [T]) -> Result<(), SparseOperatorError>,
    mut checkpoint: impl FnMut() -> Result<(), SparseOperatorError>,
) -> Result<PcgOutcome, RuntimeError> {
    let mut runtime_checkpoint =
        || checkpoint().map_err(|error| super::execution::map_operator_in_context(error, context));
    let recurrence = RecurrenceState::new(
        input.rhs,
        input.initial,
        input.tolerance,
        input.maximum_iterations,
        input.work,
        (false, false),
        &mut runtime_checkpoint,
    )?;
    drive_sync(
        recurrence,
        input.operation_scratch,
        |operation, input, output| {
            debug_assert_eq!(operation, PcgOperation::Coefficient);
            matvec(input, output)
                .map_err(|error| super::execution::map_operator_in_context(error, context))
        },
        runtime_checkpoint,
    )
}
