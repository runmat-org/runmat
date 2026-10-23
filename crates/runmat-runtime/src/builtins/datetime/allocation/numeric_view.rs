//! Borrowed native calendar numbers; exact integer storage is checked before f64 conversion.
use crate::builtins::common::tensor;
use crate::builtins::datetime::{conversion::CalendarWork, datetime_error};
use crate::BuiltinResult;
use runmat_value::{LogicalArray, Tensor, Value};

#[derive(Clone, Copy)]
pub(in crate::builtins::datetime) enum NumericView<'a> {
    Scalar(f64),
    Tensor(&'a Tensor),
    Logical(&'a LogicalArray),
}

impl<'a> NumericView<'a> {
    pub(in crate::builtins::datetime) async fn borrow(
        value: &'a Value,
        label: &str,
        serial: bool,
        work: &mut CalendarWork<'_>,
    ) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self::Scalar(*value)),
            Value::Bool(value) => Ok(Self::Scalar(f64::from(u8::from(*value)))),
            Value::Int(value) => {
                if !(-1_000_000_000..=1_000_000_000).contains(&value.to_i128()) {
                    let name = if serial {
                        "integer serial"
                    } else {
                        "integer value"
                    };
                    return Err(datetime_error(format!(
                        "datetime: {label} {name} is outside the supported calendar range"
                    )));
                }
                Ok(Self::Scalar(value.to_f64()))
            }
            Value::Tensor(tensor) => {
                validate_integer_prefix(tensor, tensor.len(), label, serial, work).await?;
                Ok(Self::Tensor(tensor))
            }
            Value::LogicalArray(array) => Ok(Self::Logical(array)),
            _ => Err(datetime_error(format!(
                "datetime: {label}: expected numeric calendar input"
            ))),
        }
    }

    pub(in crate::builtins::datetime) fn len(self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.len(),
            Self::Logical(array) => array.len(),
        }
    }

    pub(in crate::builtins::datetime) fn shape(self) -> &'a [usize] {
        const SCALAR: &[usize] = &[1, 1];
        match self {
            Self::Scalar(_) => SCALAR,
            Self::Tensor(tensor) if tensor.shape().is_empty() => SCALAR,
            Self::Tensor(tensor) => tensor.shape(),
            Self::Logical(array) if array.shape.is_empty() => SCALAR,
            Self::Logical(array) => &array.shape,
        }
    }

    pub(in crate::builtins::datetime) fn at(self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => value,
            Self::Tensor(tensor) => tensor::tensor_value_f64(tensor, index),
            Self::Logical(array) => f64::from(u8::from(array.data.as_slice()[index] != 0)),
        }
    }
}

pub(in crate::builtins::datetime) async fn validate_integer_prefix(
    tensor: &Tensor,
    count: usize,
    label: &str,
    serial: bool,
    work: &mut CalendarWork<'_>,
) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for index in 0..count {
        let value = storage.value_at(index).ok_or_else(|| {
            datetime_error("datetime: calendar integer storage is shorter than its shape")
        })?;
        if value
            .try_to_i64()
            .is_none_or(|value| !(-1_000_000_000..=1_000_000_000).contains(&value))
        {
            let name = if serial {
                "integer serial"
            } else {
                "integer value"
            };
            return Err(datetime_error(format!(
                "datetime: {label} {name} is outside the supported calendar range"
            )));
        }
        work.charge(1).await?;
    }
    work.flush().await
}
