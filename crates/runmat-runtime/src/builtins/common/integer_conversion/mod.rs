mod class;
mod error;
mod host;
mod provider;
mod representations;
mod storage;

pub(crate) use class::IntegerClassExt;
pub(crate) use error::CastError;
pub(crate) use representations::cast_complex_value;
pub(crate) use runmat_types::IntegerClass;
use runmat_value::Value;
pub(crate) use storage::integer_values;

pub(crate) async fn cast_value(value: Value, target: IntegerClass) -> Result<Value, CastError> {
    match value {
        Value::GpuTensor(handle) => provider::cast_gpu_value(handle, target).await,
        value => host::cast_host_value(value, target),
    }
}

#[cfg(test)]
mod tests;
