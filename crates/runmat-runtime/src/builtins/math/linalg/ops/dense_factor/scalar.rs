use crate::context::host_allocation::HostTypedElement;
use faer::traits::ComplexField;
use faer::{c32, c64};
use runmat_value::{ComplexElement, Value};

pub(in crate::builtins::math::linalg) trait FaerFactorScalar:
    super::super::linear_operator::NativeScalarView
{
    type Element: HostTypedElement + ComplexField + Send + 'static;
    fn source(value: &Value) -> Option<&[Self]>;
    fn to_factor(self) -> Self::Element;
    fn from_factor(value: Self::Element) -> Option<Self>;
    fn finite(value: Self::Element) -> bool;
}

macro_rules! real_scalar {
    ($scalar:ty, $view:ident) => {
        impl FaerFactorScalar for $scalar {
            type Element = $scalar;
            fn source(value: &Value) -> Option<&[Self]> {
                let Value::Tensor(tensor) = value else {
                    return None;
                };
                tensor.$view()
            }
            fn to_factor(self) -> Self::Element {
                self
            }
            fn from_factor(value: Self::Element) -> Option<Self> {
                value.is_finite().then_some(value)
            }
            fn finite(value: Self::Element) -> bool {
                value.is_finite()
            }
        }
    };
}

real_scalar!(f32, as_f32_slice);
real_scalar!(f64, as_f64_slice);

macro_rules! complex_scalar {
    ($scalar:ty, $element:ty, $view:ident) => {
        impl FaerFactorScalar for ComplexElement<$scalar> {
            type Element = $element;
            fn source(value: &Value) -> Option<&[Self]> {
                let Value::ComplexTensor(tensor) = value else {
                    return None;
                };
                tensor.$view()
            }
            fn to_factor(self) -> Self::Element {
                <$element>::new(self.0, self.1)
            }
            fn from_factor(value: Self::Element) -> Option<Self> {
                (value.re.is_finite() && value.im.is_finite())
                    .then_some(ComplexElement(value.re, value.im))
            }
            fn finite(value: Self::Element) -> bool {
                value.re.is_finite() && value.im.is_finite()
            }
        }
    };
}

complex_scalar!(f32, c32, as_f32_slice);
complex_scalar!(f64, c64, as_f64_slice);
