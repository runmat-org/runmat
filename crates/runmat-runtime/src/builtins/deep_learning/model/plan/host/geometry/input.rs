//! Input payload metadata and the input layer's axis interpretation.

use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{DeepLearningInputKind, DeepLearningLayerKind as K};
use runmat_value::{NumericDType, ObjectInstance, Value};

use crate::builtins::deep_learning::model::admission::forward_data;
use crate::builtins::deep_learning::{deep_learning_error, numeric_vector, DLARRAY_CLASS};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) fn metadata<'a>(
    input: &'a Value,
    function: &'static str,
) -> BuiltinResult<(&'a [usize], NumericDType, String)> {
    let format = match input {
        Value::Object(object) if object.class_name.is(DLARRAY_CLASS) => {
            match object.properties.get("Format") {
                Some(Value::String(value)) => value.clone(),
                None => String::new(),
                _ => return Err(fail(function, "dlarray Format must be text")),
            }
        }
        _ => String::new(),
    };
    let data = forward_data(input);
    let (shape, dtype) = match data {
        Value::Tensor(tensor) => (
            tensor.shape.as_slice(),
            if tensor.numeric_dtype().integer_class().is_some() {
                // Preserve the stored integer input and convert only at the
                // declared floating execution boundary.
                NumericDType::F64
            } else {
                tensor.numeric_dtype()
            },
        ),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| fail(function, error))?;
            if crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle).is_none() {
                return Err(fail(function, "input has no live owning provider"));
            }
            let dtype = match runmat_accelerate_api::handle_precision(handle) {
                Some(ProviderPrecision::F32) => NumericDType::F32,
                Some(ProviderPrecision::F64) => NumericDType::F64,
                None => return Err(fail(function, "input has unsupported resident precision")),
            };
            (handle.shape(), dtype)
        }
        Value::LogicalArray(array) => (array.shape.as_slice(), NumericDType::F64),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (&[1usize, 1][..], NumericDType::F64),
        _ => return Err(fail(function, "input must contain real numeric data")),
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(fail(
            function,
            "host network input requires single or double precision",
        ));
    }
    Ok((shape, dtype, format))
}

pub(super) fn kind(
    first: &ObjectInstance,
    first_kind: K,
    function: &'static str,
) -> BuiltinResult<DeepLearningInputKind> {
    let value = first.properties.get("InputSize").ok_or_else(|| {
        fail(
            function,
            if first_kind == K::FeatureInput {
                "featureInputLayer is missing InputSize"
            } else {
                "sequenceInputLayer is missing InputSize"
            },
        )
    })?;
    let vector = numeric_vector(value, function, "InputSize")?;
    Ok(match (first_kind, vector.len() == 1) {
        (K::FeatureInput, true) => DeepLearningInputKind::Feature,
        (K::FeatureInput, false) => DeepLearningInputKind::Spatial1d,
        (K::SequenceInput, true) => DeepLearningInputKind::VectorSequence,
        (K::SequenceInput, false) => DeepLearningInputKind::SpatialSequence1d,
        _ => return Err(fail(function, "network must start with an input layer")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{NumericStorage, Tensor};

    #[test]
    fn all_integer_predictor_classes_enter_the_double_boundary_without_storage_mutation() {
        let storages = [
            NumericStorage::I8(vec![1, 2]),
            NumericStorage::I16(vec![1, 2]),
            NumericStorage::I32(vec![1, 2]),
            NumericStorage::I64(vec![1, 2]),
            NumericStorage::U8(vec![1, 2]),
            NumericStorage::U16(vec![1, 2]),
            NumericStorage::U32(vec![1, 2]),
            NumericStorage::U64(vec![1, 2]),
        ];
        for storage in storages {
            let tensor = Tensor::from_numeric_storage(storage, vec![1, 2]).unwrap();
            let original_dtype = tensor.numeric_dtype();
            let input = Value::Tensor(tensor);
            let (shape, precision, format) = metadata(&input, "forward").unwrap();
            assert_eq!(shape, &[1, 2]);
            assert_eq!(precision, NumericDType::F64);
            assert!(format.is_empty());
            let Value::Tensor(tensor) = input else {
                unreachable!()
            };
            assert_eq!(tensor.numeric_dtype(), original_dtype);
        }
    }
}
