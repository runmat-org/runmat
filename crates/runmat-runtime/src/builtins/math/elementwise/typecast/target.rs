use runmat_builtins::TYPECAST_ERROR_INVALID_ARGUMENT;
use runmat_types::{standard, ClassIdentity, NumericClass};
use runmat_value::{NumericDType, Value};

use crate::BuiltinResult;

use super::error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Representation {
    Numeric(NumericDType),
    Logical,
}

impl Representation {
    pub(super) fn width(self) -> usize {
        match self {
            Self::Numeric(dtype) => dtype.byte_size(),
            Self::Logical => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OutputTarget {
    pub(super) representation: Representation,
    pub(super) complex: bool,
}

impl OutputTarget {
    pub(super) fn from_selector(value: &Value) -> BuiltinResult<Self> {
        let Value::String(name) = value else {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "newtype must be a string scalar",
            ));
        };
        let identity = ClassIdentity::new(name.to_ascii_lowercase()).map_err(|_| {
            error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "newtype must name a valid class",
            )
        })?;
        let representation = if let Some(class) = NumericClass::from_class_identity(&identity) {
            Representation::Numeric(NumericDType::from(class))
        } else if identity.is(standard::LOGICAL) {
            Representation::Logical
        } else if identity.is(standard::CHAR) {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "character reinterpretation is not implemented",
            ));
        } else {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "unsupported output class",
            ));
        };
        Ok(Self {
            representation,
            complex: false,
        })
    }

    pub(super) fn from_prototype(
        source: &Value,
        selector: &Value,
        prototype: &Value,
    ) -> BuiltinResult<Self> {
        if !matches!(selector, Value::String(keyword) if keyword.eq_ignore_ascii_case("like")) {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "three-input syntax requires the literal string \"like\"",
            ));
        }
        if matches!(source, Value::GpuTensor(_)) || matches!(prototype, Value::GpuTensor(_)) {
            return Err(error::terminal_gpu(
                "gpuArray input does not support the like syntax",
            ));
        }
        from_value(prototype)
    }
}

fn from_value(value: &Value) -> BuiltinResult<OutputTarget> {
    let (representation, complex) = match value {
        Value::Num(_) => (Representation::Numeric(NumericDType::F64), false),
        Value::Complex(_, _) => (Representation::Numeric(NumericDType::F64), true),
        Value::Int(value) => (Representation::Numeric(value.numeric_dtype()), false),
        Value::Bool(_) | Value::LogicalArray(_) => (Representation::Logical, false),
        Value::Tensor(tensor) => (Representation::Numeric(tensor.numeric_dtype()), false),
        Value::ComplexTensor(tensor) => (Representation::Numeric(tensor.numeric_dtype()), true),
        _ => {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "unsupported like prototype",
            ));
        }
    };
    Ok(OutputTarget {
        representation,
        complex,
    })
}
