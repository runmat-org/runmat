use crate::builtins::common::uniform_scalar_output::{UniformScalarCollector, UniformScalarError};
use crate::execution::{RootedValue, RootedValuesBuilder};
use runmat_value::{StructValue, Value};

use super::error;

pub(super) enum Collector {
    Uniform(UniformScalarCollector),
    Structure {
        names: Vec<String>,
        values: RootedValuesBuilder,
    },
}

impl Collector {
    pub(super) fn new(uniform: bool) -> Self {
        if uniform {
            Self::Uniform(UniformScalarCollector::new())
        } else {
            Self::Structure {
                names: Vec::new(),
                values: RootedValuesBuilder::default(),
            }
        }
    }

    pub(super) fn push(&mut self, field: &str, value: RootedValue) -> crate::BuiltinResult<()> {
        match self {
            Self::Uniform(collector) => collector.push(value.value()).map_err(map_uniform_error),
            Self::Structure { names, values } => {
                value
                    .with_owned(|value| values.push(value))
                    .map_err(|error| error.into_runtime_error())?;
                names.push(field.to_string());
                Ok(())
            }
        }
    }

    pub(super) fn finish(self, field_count: usize) -> crate::BuiltinResult<RootedValue> {
        match self {
            Self::Uniform(collector) => collector
                .finish(&[field_count, 1])
                .map_err(map_uniform_error)
                .and_then(|value| {
                    RootedValue::new(value).map_err(|error| error.into_runtime_error())
                }),
            Self::Structure { names, values } => values
                .finish()
                .map_err(|error| error.into_runtime_error())?
                .with_unbounded_values(|values| {
                    let mut structure = StructValue::new();
                    for (name, value) in names.into_iter().zip(values) {
                        structure.insert(&name, value);
                    }
                    RootedValue::new(Value::Struct(structure))
                        .map_err(|error| error.into_runtime_error())
                })?,
        }
    }
}

pub(super) fn normalize(
    value: RootedValue,
    requested: usize,
) -> crate::BuiltinResult<Vec<RootedValue>> {
    value.with_owned(|value| {
        normalize_raw(value, requested)?
            .try_into_owned()
            .map_err(|_| {
                error::function("structfun: shared callback outputs require separate admission")
            })?
            .into_iter()
            .map(|value| RootedValue::new(value).map_err(|error| error.into_runtime_error()))
            .collect()
    })
}

fn normalize_raw(
    value: Value,
    requested: usize,
) -> crate::BuiltinResult<runmat_value::OutputListStorage> {
    match value {
        Value::OutputList(values) if values.len() == requested => Ok(values),
        Value::OutputList(values) => Err(error::function(format!(
            "structfun: callback returned {} outputs but {requested} were requested",
            values.len()
        ))),
        value if requested == 1 => Ok(runmat_value::OutputListStorage::unbounded(vec![value])),
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
