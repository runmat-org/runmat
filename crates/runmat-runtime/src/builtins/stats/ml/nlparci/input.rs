use std::borrow::Cow;

use runmat_value::{NumericDType, Tensor, Value};

use super::*;
use crate::builtins::common::tensor;
use crate::gather_if_needed_async;

/// A host scalar is already a complete numeric input. Preserve that borrowed
/// payload instead of constructing a one-element tensor after finite admission.
pub(super) enum HostInput {
    Scalar(f64),
    Tensor(Tensor),
}

impl HostInput {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.len(),
        }
    }

    pub(super) fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::Scalar(_) => NumericDType::F64,
            Self::Tensor(tensor) => tensor.numeric_dtype(),
        }
    }

    pub(super) fn value_f64(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => {
                debug_assert_eq!(index, 0);
                *value
            }
            Self::Tensor(tensor) => tensor::tensor_value_f64(tensor, index),
        }
    }

    pub(super) fn values_f64_cow(&self) -> Cow<'_, [f64]> {
        match self {
            Self::Scalar(value) => Cow::Borrowed(std::slice::from_ref(value)),
            Self::Tensor(tensor) => tensor::tensor_values_f64_cow(tensor),
        }
    }
}

pub(super) async fn host_input(source: Value) -> BuiltinResult<HostInput> {
    let source = match source {
        Value::Num(value) => return Ok(HostInput::Scalar(value)),
        other => other,
    };
    let value = gather_if_needed_async(&source).await?;
    match value {
        Value::Num(value) => Ok(HostInput::Scalar(value)),
        other => tensor::value_into_tensor_for(NAME, other)
            .map(HostInput::Tensor)
            .map_err(|err| {
                invalid(format!(
                    "nlparci requires real floating numeric input: {err}"
                ))
            }),
    }
}
