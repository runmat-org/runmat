//! Inline scalar views share native classes with array operators. Complex
//! scalar fields are copied into one typed element, never reinterpreted.
use super::{scalar::LinearOperatorScalar, LinearOperatorError};
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use runmat_value::{ComplexElement, Value};

pub(in crate::builtins::math::linalg) trait NativeScalarView:
    Copy
{
    fn scalar(value: &Value) -> Option<Self>;
}
impl NativeScalarView for f32 {
    fn scalar(_: &Value) -> Option<Self> {
        None
    }
}
impl NativeScalarView for ComplexElement<f32> {
    fn scalar(_: &Value) -> Option<Self> {
        None
    }
}
impl NativeScalarView for f64 {
    fn scalar(value: &Value) -> Option<Self> {
        match value {
            Value::Num(value) => Some(*value),
            _ => None,
        }
    }
}
impl NativeScalarView for ComplexElement<f64> {
    fn scalar(value: &Value) -> Option<Self> {
        match value {
            Value::Complex(real, imaginary) => Some(ComplexElement(*real, *imaginary)),
            _ => None,
        }
    }
}

pub(super) fn product<T: LinearOperatorScalar>(
    value: T,
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    adjoint: bool,
) -> Result<(), LinearOperatorError> {
    if input.len() != 1 || output.len() != 1 {
        return Err(LinearOperatorError::DimensionMismatch);
    }
    HostAllocationReceipt::policy(context)?;
    output[0] = T::zero();
    T::add_product(
        &mut output[0],
        if adjoint { value.conjugate() } else { value },
        input[0],
    );
    HostAllocationReceipt::policy(context)?;
    Ok(())
}
