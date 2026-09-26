use crate::builtins::common::uniform_scalar_output::UniformScalarCollector;
use crate::{gather_if_needed_async, make_cell_with_shape, BuiltinResult};
use runmat_value::Value;

pub(in crate::builtins::cells::core::cellfun) enum Collector {
    Uniform(UniformScalarCollector),
    Cell(Vec<Value>),
}
impl Collector {
    pub(in crate::builtins::cells::core::cellfun) fn new(uniform: bool) -> Self {
        if uniform {
            Self::Uniform(UniformScalarCollector::new())
        } else {
            Self::Cell(Vec::new())
        }
    }
    pub(in crate::builtins::cells::core::cellfun) async fn push(
        &mut self,
        value: Value,
    ) -> BuiltinResult<()> {
        match self {
            Self::Uniform(collector) => collector
                .push(&gather_if_needed_async(&value).await?)
                .map_err(super::map_error),
            Self::Cell(values) => {
                values.push(value);
                Ok(())
            }
        }
    }
    pub(in crate::builtins::cells::core::cellfun) fn finish(
        self,
        shape: &[usize],
    ) -> BuiltinResult<Value> {
        match self {
            Self::Uniform(collector) => collector.finish(shape).map_err(super::map_error),
            Self::Cell(values) => make_cell_with_shape(values, shape.to_vec())
                .map_err(|reason| super::super::error::internal(format!("cellfun: {reason}"))),
        }
    }
}
pub(in crate::builtins::cells::core::cellfun) fn normalize(
    value: Value,
    count: usize,
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == count => Ok(values),
        value if count == 1 && !matches!(value, Value::OutputList(_)) => Ok(vec![value]),
        _ => Err(super::super::error::invalid(
            "cellfun: callback did not return the requested number of outputs",
        )),
    }
}
