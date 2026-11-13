//! Native scalar policy and matrix adapter for the shared PCG owner.

use runmat_value::ComplexElement;

use super::async_driver::AsyncPcgOperation;
use super::async_driver::PcgOperation;
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use crate::RuntimeError;

pub(super) trait PcgScalar: Copy + PartialEq {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain;
    const HERMITIAN_EPSILON: f64;
    fn zero() -> Self;
    fn add_scaled(&mut self, scale: f64, value: Self);
    fn subtract_scaled(&mut self, scale: f64, value: Self);
    fn components(self) -> (f64, f64);
    fn scaled_conjugate_product(self, other: Self, left: f64, right: f64) -> (f64, f64);
}

macro_rules! real_scalar {
    ($scalar:ty) => {
        impl PcgScalar for $scalar {
            const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
                runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Real;
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
            fn components(self) -> (f64, f64) {
                (f64::from(self), 0.0)
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
            const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
                runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Complex;
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
            fn components(self) -> (f64, f64) {
                (f64::from(self.0), f64::from(self.1))
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

pub(super) use runmat_sparse_solvers::krylov::pcg::{PcgOutcome, PcgWork};
pub(super) struct Native<T>(std::marker::PhantomData<T>);
impl<T: PcgScalar> runmat_sparse_solvers::krylov::pcg::Arithmetic<T> for Native<T> {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain = T::DOMAIN;
    const HERMITIAN_EPSILON: f64 = T::HERMITIAN_EPSILON;
    fn zero() -> T {
        T::zero()
    }
    fn components(value: T) -> (f64, f64) {
        value.components()
    }
    fn add_scaled(target: &mut T, scale: f64, value: T) {
        target.add_scaled(scale, value);
    }
    fn subtract_scaled(target: &mut T, scale: f64, value: T) {
        target.subtract_scaled(scale, value);
    }
    fn scaled_conjugate_product(
        left: T,
        right: T,
        left_scale: f64,
        right_scale: f64,
    ) -> (f64, f64) {
        left.scaled_conjugate_product(right, left_scale, right_scale)
    }
}

pub(super) struct PcgSolveInput<'a, T: PcgScalar> {
    pub(super) rhs: &'a [T],
    pub(super) initial: Option<&'a [T]>,
    pub(super) tolerance: f64,
    pub(super) maximum_iterations: usize,
    pub(super) work: PcgWork<'a, T>,
    pub(super) operation_scratch: &'a mut [T],
}

struct MatrixOperation<'a, M, C> {
    matvec: M,
    checkpoint: C,
    context: Option<&'a RuntimeContext>,
}
impl<
        T: PcgScalar,
        M: FnMut(&[T], &mut [T]) -> Result<(), SparseOperatorError>,
        C: FnMut() -> Result<(), SparseOperatorError>,
    > AsyncPcgOperation<T> for MatrixOperation<'_, M, C>
{
    fn checkpoint(&mut self) -> Result<(), RuntimeError> {
        (self.checkpoint)()
            .map_err(|error| super::execution::map_operator_in_context(error, self.context))
    }
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
        context: Option<&RuntimeContext>,
    ) -> Result<super::super::iterative::PreconditionerStatus, RuntimeError> {
        debug_assert_eq!(operation, PcgOperation::Coefficient);
        (self.matvec)(input, output)
            .map_err(|error| super::execution::map_operator_in_context(error, context))?;
        Ok(super::super::iterative::PreconditionerStatus::Applied)
    }
}
pub(super) async fn solve<T: PcgScalar>(
    input: PcgSolveInput<'_, T>,
    context: Option<&RuntimeContext>,
    matvec: impl FnMut(&[T], &mut [T]) -> Result<(), SparseOperatorError>,
    checkpoint: impl FnMut() -> Result<(), SparseOperatorError>,
    receipt: Option<&HostAllocationReceipt>,
) -> Result<PcgOutcome, RuntimeError> {
    super::async_driver::drive(
        input,
        MatrixOperation {
            matvec,
            checkpoint,
            context,
        },
        context,
        receipt,
        (false, false),
    )
    .await
}
