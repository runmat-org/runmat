//! Syntax admission and numeric materialization are separate provider boundaries.

use super::*;
use crate::builtins::common::random_args::keyword_of;
use crate::gather_if_needed_async;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_builtins::NormalizeKeyword;

mod admission;
mod input;
mod materialize;

pub(super) use admission::admit_args;
use input::ensure_exact_integer_value;
pub(super) use input::value_to_input;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum InputDomain {
    Real,
    Complex,
}

impl InputDomain {
    pub(super) fn of(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_) => {
                ensure_exact_integer_value(value, "input data")?;
                Ok(Self::Real)
            }
            Value::Complex(..) | Value::ComplexTensor(_) => {
                ensure_exact_integer_value(value, "input data")?;
                Ok(Self::Complex)
            }
            Value::GpuTensor(handle) => match runmat_accelerate_api::handle_storage(handle) {
                GpuTensorStorage::Real => Ok(Self::Real),
                GpuTensorStorage::ComplexInterleaved => Ok(Self::Complex),
            },
            other => Err(normalize_error(format!(
                "normalize: unsupported input type {other:?}"
            ))),
        }
    }
}

enum CenterArg {
    Mean,
    Median,
    Explicit(Value),
}

enum ScaleArg {
    Std,
    Mad,
    Iqr,
    First,
    Explicit(Value),
}

enum Method {
    ZScore {
        robust: bool,
    },
    Norm(Option<Value>),
    Scale(ScaleArg),
    Center {
        center: CenterArg,
        scale: Option<ScaleArg>,
    },
    MedianIqr,
    Range(Option<Value>),
}

pub(super) struct AdmittedArgs {
    dimension: Option<Value>,
    method: Method,
}
