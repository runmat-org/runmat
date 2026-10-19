use runmat_builtins::{BESSELJ_ERROR_INVALID_ARGUMENT, BESSELJ_ERROR_INVALID_INPUT};
use runmat_value::{ComplexElement, NumericDType, Value};

use crate::BuiltinResult;

use super::error;

pub(super) enum Order<'a> {
    Scalar(f64),
    Double(&'a [f64], &'a [usize]),
    Single(&'a [f32], &'a [usize]),
}

pub(super) enum Argument<'a> {
    RealScalar(f64),
    ComplexScalar(f64, f64),
    Double(&'a [f64], &'a [usize]),
    Single(&'a [f32], &'a [usize]),
    ComplexDouble(&'a [ComplexElement<f64>], &'a [usize]),
    ComplexSingle(&'a [ComplexElement<f32>], &'a [usize]),
}

impl<'a> Order<'a> {
    pub(super) fn parse(value: &'a Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(number) => Ok(Self::Scalar(*number)),
            Value::Tensor(tensor) => match tensor.numeric_dtype() {
                NumericDType::F64 => Ok(Self::Double(
                    tensor.as_f64_slice().expect("double storage"),
                    tensor.shape(),
                )),
                NumericDType::F32 => Ok(Self::Single(
                    tensor.as_f32_slice().expect("single storage"),
                    tensor.shape(),
                )),
                _ => Err(invalid("nu must be real single or double")),
            },
            _ => Err(invalid("nu must be real single or double")),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Double(values, _) => values.len(),
            Self::Single(values, _) => values.len(),
        }
    }

    pub(super) fn shape(&self) -> Option<&'a [usize]> {
        match self {
            Self::Scalar(_) => None,
            Self::Double(_, shape) | Self::Single(_, shape) => Some(shape),
        }
    }

    pub(super) fn is_single(&self) -> bool {
        matches!(self, Self::Single(..))
    }

    pub(super) fn get(&self, index: usize) -> f64 {
        let index = if self.len() == 1 { 0 } else { index };
        match self {
            Self::Scalar(number) => *number,
            Self::Double(values, _) => values[index],
            Self::Single(values, _) => f64::from(values[index]),
        }
    }
}

impl<'a> Argument<'a> {
    pub(super) fn parse(value: &'a Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(number) => Ok(Self::RealScalar(*number)),
            Value::Complex(real, imaginary) => Ok(Self::ComplexScalar(*real, *imaginary)),
            Value::Tensor(tensor) => match tensor.numeric_dtype() {
                NumericDType::F64 => Ok(Self::Double(
                    tensor.as_f64_slice().expect("double storage"),
                    tensor.shape(),
                )),
                NumericDType::F32 => Ok(Self::Single(
                    tensor.as_f32_slice().expect("single storage"),
                    tensor.shape(),
                )),
                _ => Err(invalid("Z must be real or complex single/double")),
            },
            Value::ComplexTensor(tensor) => match tensor.numeric_dtype() {
                NumericDType::F64 => Ok(Self::ComplexDouble(
                    tensor.as_f64_slice().expect("complex double storage"),
                    tensor.shape(),
                )),
                NumericDType::F32 => Ok(Self::ComplexSingle(
                    tensor.as_f32_slice().expect("complex single storage"),
                    tensor.shape(),
                )),
                _ => Err(invalid("Z must be real or complex single/double")),
            },
            _ => Err(invalid("Z must be real or complex single/double")),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::RealScalar(..) | Self::ComplexScalar(..) => 1,
            Self::Double(values, _) => values.len(),
            Self::Single(values, _) => values.len(),
            Self::ComplexDouble(values, _) => values.len(),
            Self::ComplexSingle(values, _) => values.len(),
        }
    }

    pub(super) fn shape(&self) -> Option<&'a [usize]> {
        match self {
            Self::RealScalar(..) | Self::ComplexScalar(..) => None,
            Self::Double(_, shape)
            | Self::Single(_, shape)
            | Self::ComplexDouble(_, shape)
            | Self::ComplexSingle(_, shape) => Some(shape),
        }
    }

    pub(super) fn is_single(&self) -> bool {
        matches!(self, Self::Single(..) | Self::ComplexSingle(..))
    }

    pub(super) fn is_complex(&self) -> bool {
        matches!(
            self,
            Self::ComplexScalar(..) | Self::ComplexDouble(..) | Self::ComplexSingle(..)
        )
    }

    pub(super) fn get(&self, index: usize) -> (f64, f64) {
        let index = if self.len() == 1 { 0 } else { index };
        match self {
            Self::RealScalar(real) => (*real, 0.0),
            Self::ComplexScalar(real, imag) => (*real, *imag),
            Self::Double(values, _) => (values[index], 0.0),
            Self::Single(values, _) => (f64::from(values[index]), 0.0),
            Self::ComplexDouble(values, _) => (values[index].0, values[index].1),
            Self::ComplexSingle(values, _) => {
                (f64::from(values[index].0), f64::from(values[index].1))
            }
        }
    }
}

pub(super) fn shape<'a>(order: &Order<'a>, argument: &Argument<'a>) -> Option<&'a [usize]> {
    let order_shape = order.shape();
    let argument_shape = argument.shape();
    if order.len() == 1 && argument.len() == 1 {
        return Some(argument_shape.or(order_shape).unwrap_or(&[1, 1]));
    }
    if order.len() == 1 {
        return argument_shape;
    }
    if argument.len() == 1 {
        return order_shape;
    }
    if order_shape == argument_shape {
        argument_shape
    } else {
        None
    }
}

pub(super) fn scale(value: &Value) -> BuiltinResult<bool> {
    let number = match value {
        Value::Num(number) => *number,
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_dtype() {
            NumericDType::F64 => tensor.as_f64_slice().expect("double storage")[0],
            NumericDType::F32 => f64::from(tensor.as_f32_slice().expect("single storage")[0]),
            _ => {
                return Err(error(
                    &BESSELJ_ERROR_INVALID_ARGUMENT,
                    "scale must be 0 or 1",
                ))
            }
        },
        _ => {
            return Err(error(
                &BESSELJ_ERROR_INVALID_ARGUMENT,
                "scale must be 0 or 1",
            ))
        }
    };
    match number {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(error(
            &BESSELJ_ERROR_INVALID_ARGUMENT,
            "scale must be 0 or 1",
        )),
    }
}

fn invalid(detail: &'static str) -> crate::RuntimeError {
    error(&BESSELJ_ERROR_INVALID_INPUT, detail)
}
