use super::super::super::{
    capabilities::{DATETIME_CLASS, SERIAL_FIELD},
    datetime_error,
};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};

/// Borrow the existing object generation; never manufacture scalar storage or
/// materialize a native numeric payload merely to enter the calendar domain.
pub(in crate::builtins::datetime) enum SerialView<'a> {
    Scalar(f64),
    Tensor(&'a Tensor),
}

impl<'a> SerialView<'a> {
    pub(in crate::builtins::datetime) fn borrow(value: &'a Value) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(datetime_error("datetime: expected a datetime value"));
        };
        if !object.is_class(DATETIME_CLASS) {
            return Err(datetime_error("datetime: expected a datetime value"));
        }
        match object.property(SERIAL_FIELD) {
            Some(Value::Num(value)) => Ok(Self::Scalar(*value)),
            Some(Value::Tensor(tensor)) => Ok(Self::Tensor(tensor)),
            Some(other) => Err(datetime_error(format!(
                "datetime: invalid internal serial storage {other:?}"
            ))),
            None => Err(datetime_error("datetime: missing internal serial storage")),
        }
    }

    pub(in crate::builtins::datetime) fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.len(),
        }
    }

    pub(in crate::builtins::datetime) fn shape(&self) -> &'a [usize] {
        const SCALAR: &[usize] = &[1, 1];
        match self {
            Self::Scalar(_) => SCALAR,
            Self::Tensor(tensor) => tensor.shape(),
        }
    }

    pub(in crate::builtins::datetime) fn at(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            // Exactly the existing per-native-element conversion, without the
            // Cow materialization used by the older component helper.
            Self::Tensor(tensor) => {
                crate::builtins::common::tensor::tensor_value_f64(tensor, index)
            }
        }
    }
}
