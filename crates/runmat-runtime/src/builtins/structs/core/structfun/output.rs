use crate::builtins::common::uniform_scalar_output::{UniformScalarCollector, UniformScalarError};
use runmat_value::{StructValue, Value};

use super::error;

pub(super) enum Collector {
    Uniform(UniformScalarCollector),
    Structure(StructValue),
}

impl Collector {
    pub(super) fn new(uniform: bool) -> Self {
        if uniform {
            Self::Uniform(UniformScalarCollector::new())
        } else {
            Self::Structure(StructValue::new())
        }
    }

    pub(super) fn push(&mut self, field: &str, value: Value) -> crate::BuiltinResult<()> {
        match self {
            Self::Uniform(collector) => collector.push(&value).map_err(map_uniform_error),
            Self::Structure(output) => {
                output.insert(field, value);
                Ok(())
            }
        }
    }

    pub(super) fn finish(self, field_count: usize) -> crate::BuiltinResult<Value> {
        match self {
            Self::Uniform(collector) => collector
                .finish(&[field_count, 1])
                .map_err(map_uniform_error),
            Self::Structure(output) => Ok(Value::Struct(output)),
        }
    }
}

pub(super) fn normalize(value: Value, requested: usize) -> crate::BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == requested => Ok(values),
        Value::OutputList(values) => Err(error::function(format!(
            "structfun: callback returned {} outputs but {requested} were requested",
            values.len()
        ))),
        value if requested == 1 => Ok(vec![value]),
        _ => Err(error::function(
            "structfun: callback did not return the requested number of outputs",
        )),
    }
}

fn map_uniform_error(value: UniformScalarError) -> crate::RuntimeError {
    match value {
        UniformScalarError::Materialization(reason) => error::internal(format!("structfun: {reason}")),
        UniformScalarError::InvalidStorage(kind) => error::internal(format!("structfun: {kind} has no storage value")),
        UniformScalarError::CharacterRank => error::uniform("structfun: uniform character outputs must form a 2-D column"),
        UniformScalarError::CharacterCount => error::uniform("structfun: callback returned the wrong number of characters"),
        UniformScalarError::SizeOverflow => error::internal("structfun: output size exceeds platform limits"),
        UniformScalarError::NonScalar => error::uniform("structfun: callback must return scalar numeric, logical, character, or complex values when UniformOutput is true"),
        UniformScalarError::Heterogeneous => error::uniform("structfun: callback outputs with UniformOutput=true must have the same data type on every invocation"),
    }
}
