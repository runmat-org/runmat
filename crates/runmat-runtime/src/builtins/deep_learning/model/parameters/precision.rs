//! Borrowed floating precision of admitted learnable payloads.
use super::*;

pub(in crate::builtins::deep_learning) fn floating_precision(
    value: &Value,
) -> Option<NumericDType> {
    match value {
        Value::Tensor(tensor) if !tensor.shape().contains(&0) => Some(tensor.numeric_dtype()),
        Value::GpuTensor(handle) if !handle.shape().contains(&0) => {
            runmat_accelerate_api::handle_precision(handle).map(|precision| match precision {
                ProviderPrecision::F32 => NumericDType::F32,
                ProviderPrecision::F64 => NumericDType::F64,
            })
        }
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            object.properties().get("Data").and_then(floating_precision)
        }
        Value::Num(_) => Some(NumericDType::F64),
        _ => None,
    }
}
