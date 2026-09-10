use runmat_value::{StructArray, StructValue, Value};

pub(super) enum Target {
    Scalar(StructValue),
    Array(StructArray),
}

impl Target {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Struct(structure) => Ok(Self::Scalar(structure)),
            Value::StructArray(array) => Ok(Self::Array(array)),
            other => Err(super::error::invalid_target(&other)),
        }
    }

    pub(super) fn into_value(self) -> crate::BuiltinResult<Value> {
        match self {
            Self::Scalar(structure) => Ok(Value::Struct(structure)),
            Self::Array(array) => Ok(Value::StructArray(array)),
        }
    }
}
