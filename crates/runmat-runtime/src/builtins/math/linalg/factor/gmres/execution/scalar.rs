//! Native-class coefficient projection for the private GMRES router.

use runmat_value::{ComplexElement, Value};

use super::super::super::super::ops::iterative_preconditioner::{
    MatrixFactorError, MatrixFactorValue,
};
use super::super::super::super::ops::linear_operator::{LinearOperator, TypedLinearOperator};
use crate::context::RuntimeContext;

pub(super) trait GmresValue: MatrixFactorValue {
    fn borrow_coefficient<'a>(
        value: &'a Value,
        context: Option<&RuntimeContext>,
    ) -> Result<TypedLinearOperator<'a, Self>, MatrixFactorError>;
}

macro_rules! native_class {
    ($scalar:ty, $variant:ident) => {
        impl GmresValue for $scalar {
            fn borrow_coefficient<'a>(
                value: &'a Value,
                context: Option<&RuntimeContext>,
            ) -> Result<TypedLinearOperator<'a, Self>, MatrixFactorError> {
                let operator = match context {
                    Some(context) => LinearOperator::borrow_with_context(value, context)?,
                    None => LinearOperator::borrow_unbounded(value)?,
                };
                let LinearOperator::$variant(operator) = operator else {
                    return Err(MatrixFactorError::UnsupportedClass);
                };
                Ok(operator)
            }
        }
    };
}

native_class!(f32, RealF32);
native_class!(f64, RealF64);
native_class!(ComplexElement<f32>, ComplexF32);
native_class!(ComplexElement<f64>, ComplexF64);
