//! Native recurrence class; conversion is explicit at borrowed boundaries.
use crate::builtins::math::linalg::{
    factor::iterative::value::IterativeValue,
    ops::{
        iterative_preconditioner::MatrixFactorValue, linear_operator::scalar::LinearOperatorScalar,
        triangular_operator::TriangularScalar,
    },
};
use num_complex::Complex64;
use runmat_value::ComplexElement;
pub(super) trait Scalar:
    IterativeValue + MatrixFactorValue + LinearOperatorScalar + TriangularScalar + Send + 'static
{
    const COMPLEX: bool;
    fn complex(self) -> Complex64;
    fn native(value: Complex64) -> Option<Self>;
}
macro_rules! scalar {
    ($t:ty,$complex:expr,$to:expr,$from:expr) => {
        impl Scalar for $t {
            const COMPLEX: bool = $complex;
            fn complex(self) -> Complex64 {
                ($to)(self)
            }
            fn native(value: Complex64) -> Option<Self> {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return None;
                }
                ($from)(value)
            }
        }
    };
}
scalar!(
    f32,
    false,
    |v: f32| Complex64::new(v as f64, 0.0),
    |v: Complex64| (v.im == 0.0 && v.re.abs() <= f32::MAX as f64).then_some(v.re as f32)
);
scalar!(
    f64,
    false,
    |v: f64| Complex64::new(v, 0.0),
    |v: Complex64| (v.im == 0.0).then_some(v.re)
);
scalar!(
    ComplexElement<f32>,
    true,
    |v: ComplexElement<f32>| Complex64::new(v.0 as f64, v.1 as f64),
    |v: Complex64| (v.re.abs() <= f32::MAX as f64 && v.im.abs() <= f32::MAX as f64)
        .then_some(ComplexElement(v.re as f32, v.im as f32))
);
scalar!(
    ComplexElement<f64>,
    true,
    |v: ComplexElement<f64>| Complex64::new(v.0, v.1),
    |v: Complex64| Some(ComplexElement(v.re, v.im))
);
