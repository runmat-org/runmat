use super::*;
use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};

pub(super) fn shape(value: &Value, provider: &'static dyn AccelProvider) -> Option<[usize; 2]> {
    let precision = provider.precision();
    let dims: &[usize] = match value {
        Value::Tensor(tensor)
            if tensor.numeric_dtype()
                == match precision {
                    ProviderPrecision::F32 => NumericDType::F32,
                    ProviderPrecision::F64 => NumericDType::F64,
                } =>
        {
            if tensor
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))?
                != tensor.len()
            {
                return None;
            }
            tensor.shape()
        }
        Value::GpuTensor(handle) => {
            let element = match precision {
                ProviderPrecision::F32 => NumericElementType::F32,
                ProviderPrecision::F64 => NumericElementType::F64,
            };
            if handle.physical().storage() != GpuTensorStorage::Real
                || handle.physical().element_type() != element
                || runmat_accelerate_api::handle_is_transposed(handle)
                || !gpu_class_metadata_matches(handle, Some(precision), None, false)
                || !exact_provider_for_handle(handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            {
                return None;
            }
            handle.shape()
        }
        _ => return None,
    };
    let [rows, cols] = dims else {
        return None;
    };
    rows.checked_mul(*cols)?;
    Some([*rows, *cols])
}

pub(super) fn scalar(value: &Value, precision: ProviderPrecision) -> Option<f64> {
    let scalar = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor)
            if tensor.len() == 1
                && matches!(
                    tensor.numeric_dtype(),
                    NumericDType::F32 | NumericDType::F64
                ) =>
        {
            tensor.numeric_value_at(0)?.materialize_f64()
        }
        _ => return None,
    };
    // Controls are not uploaded or silently rounded to a different class.
    (scalar.is_finite()
        && (precision == ProviderPrecision::F64 || f64::from(scalar as f32) == scalar))
        .then_some(scalar)
}
