use crate::builtins::common::uniform_scalar_output::UniformScalarCollector;
use crate::execution::{RootedValue, RootedValuesBuilder};
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::{CellArray, Value};

pub(in crate::builtins::cells::core::cellfun) enum Collector {
    Uniform(UniformScalarCollector<'static>),
    Cell(RootedValuesBuilder),
}
impl Collector {
    pub(in crate::builtins::cells::core::cellfun) fn new(uniform: bool) -> Self {
        if uniform {
            Self::Uniform(UniformScalarCollector::new())
        } else {
            Self::Cell(RootedValuesBuilder::default())
        }
    }
    pub(in crate::builtins::cells::core::cellfun) async fn push(
        &mut self,
        value: RootedValue,
    ) -> BuiltinResult<()> {
        match self {
            Self::Uniform(collector) => collector
                .push(&gather_if_needed_async(value.value()).await?)
                .map_err(super::map_error),
            Self::Cell(values) => value
                .with_owned(|value| values.push(value))
                .map_err(|error| error.into_runtime_error()),
        }
    }
    pub(in crate::builtins::cells::core::cellfun) fn finish(
        self,
        shape: &[usize],
    ) -> BuiltinResult<RootedValue> {
        match self {
            Self::Uniform(collector) => {
                RootedValue::new(collector.finish(shape).map_err(super::map_error)?)
                    .map_err(|error| error.into_runtime_error())
            }
            Self::Cell(values) => values
                .finish()
                .map_err(|error| error.into_runtime_error())?
                .with_unbounded_values(|values| {
                    RootedValue::new(
                        CellArray::from_column_major(values, shape.to_vec())
                            .map(Value::Cell)
                            .map_err(|reason| {
                                super::super::error::internal(format!("cellfun: {reason}"))
                            })?,
                    )
                    .map_err(|error| error.into_runtime_error())
                })?,
        }
    }
}
pub(in crate::builtins::cells::core::cellfun) fn normalize(
    value: RootedValue,
    count: usize,
) -> BuiltinResult<Vec<RootedValue>> {
    value.with_owned(|value| {
        normalize_raw(value, count)?
            .into_iter()
            .map(|value| RootedValue::new(value).map_err(|error| error.into_runtime_error()))
            .collect()
    })
}

fn normalize_raw(value: Value, count: usize) -> BuiltinResult<runmat_value::OwnedOutputListValues> {
    match value {
        Value::OutputList(values) if values.len() == count => {
            values.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })
        }
        value if count == 1 && !matches!(value, Value::OutputList(_)) => {
            Ok(runmat_value::OutputListStorage::unbounded(vec![value])
                .try_into_owned()
                .expect("unique unbounded output"))
        }
        _ => Err(super::super::error::invalid(
            "cellfun: callback did not return the requested number of outputs",
        )),
    }
}
