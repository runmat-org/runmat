//! Input payload metadata and the input layer's axis interpretation.

use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    DeepLearningAxisLabel as A, DeepLearningAxisLayout, DeepLearningInputKind,
    DeepLearningLayerKind as K, DeepLearningLayerStructuralProperty as P,
};
use runmat_value::{NumericDType, ObjectInstance, Value};

use crate::builtins::deep_learning::model::admission::forward_data;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use crate::builtins::deep_learning::{deep_learning_error, numeric_vector, DLARRAY_CLASS};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning::model::plan::host) enum InputPayloadKind {
    Real,
    Complex,
}

/// A missing channel axis represents exactly one channel only on a route
/// containing global pooling. Other layer routes retain the input-layer C
/// requirement, independent of whether the data is real or complex.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ChannelPolicy {
    Required,
    SingletonForGlobalPooling,
}

impl ChannelPolicy {
    pub(super) fn for_layers(layers: &[Value]) -> Self {
        if layers.iter().skip(1).any(|layer| {
            matches!(layer, Value::Object(object)
                if K::from_identity(object.class_name()) == Some(K::GlobalAveragePooling1D))
        }) {
            Self::SingletonForGlobalPooling
        } else {
            Self::Required
        }
    }

    pub(super) const fn permits_missing_channel(self) -> bool {
        matches!(self, Self::SingletonForGlobalPooling)
    }
}

pub(super) fn metadata<'a>(
    input: &'a Value,
    function: &'static str,
) -> BuiltinResult<(&'a [usize], NumericDType, String, InputPayloadKind)> {
    let format = match input {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            match object.properties().get("Format") {
                Some(Value::String(runmat_value::StringElement::Text(value))) => value.to_string(),
                None => String::new(),
                _ => return Err(fail(function, "dlarray Format must be text")),
            }
        }
        _ => String::new(),
    };
    let data = forward_data(input);
    let (shape, dtype, payload_kind) = match data {
        Value::Tensor(tensor) => (
            tensor.shape(),
            if tensor.numeric_dtype().integer_class().is_some() {
                // Preserve the stored integer input and convert only at the
                // declared floating execution boundary.
                NumericDType::F64
            } else {
                tensor.numeric_dtype()
            },
            InputPayloadKind::Real,
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape(),
            tensor.numeric_dtype(),
            InputPayloadKind::Complex,
        ),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| fail(function, error))?;
            if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
                return Err(fail(function, "input has no live owning provider"));
            }
            let dtype = match runmat_accelerate_api::handle_precision(handle) {
                Some(ProviderPrecision::F32) => NumericDType::F32,
                Some(ProviderPrecision::F64) => NumericDType::F64,
                None => return Err(fail(function, "input has unsupported resident precision")),
            };
            (handle.shape(), dtype, InputPayloadKind::Real)
        }
        Value::LogicalArray(array) => (
            array.shape.as_slice(),
            NumericDType::F64,
            InputPayloadKind::Real,
        ),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            (&[1usize, 1][..], NumericDType::F64, InputPayloadKind::Real)
        }
        _ => return Err(fail(function, "input must contain supported numeric data")),
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(fail(
            function,
            "host network input requires real or complex single or double precision",
        ));
    }
    Ok((shape, dtype, format, payload_kind))
}

pub(super) fn axes(
    format: &str,
    kind: DeepLearningInputKind,
    shape: &[usize],
    channel_policy: ChannelPolicy,
    function: &'static str,
) -> BuiltinResult<AxisPlan> {
    if channel_policy.permits_missing_channel() && !format.is_empty() {
        let layout = DeepLearningAxisLayout::parse(format, shape.len())
            .map_err(|error| fail(function, format!("invalid input axes: {error:?}")))?;
        let axes = layout.axes();
        let no_channel = matches!(
            (kind, axes),
            (DeepLearningInputKind::VectorSequence, [A::Batch, A::Time])
                | (DeepLearningInputKind::Spatial1d, [A::Spatial, A::Batch])
                | (
                    DeepLearningInputKind::SpatialSequence1d,
                    [A::Spatial, A::Batch, A::Time]
                )
        );
        if no_channel {
            return AxisPlan::new(layout, shape)
                .map_err(|error| fail(function, format!("invalid input axes: {error:?}")));
        }
    }
    AxisPlan::for_input(format, kind, shape)
        .map_err(|error| fail(function, format!("invalid input axes: {error:?}")))
}

pub(super) fn kind(
    first: &ObjectInstance,
    first_kind: K,
    function: &'static str,
) -> BuiltinResult<DeepLearningInputKind> {
    let value = first
        .properties()
        .get(P::InputSize.canonical_name())
        .ok_or_else(|| {
            fail(
                function,
                if first_kind == K::FeatureInput {
                    "featureInputLayer is missing InputSize"
                } else {
                    "sequenceInputLayer is missing InputSize"
                },
            )
        })?;
    let vector = numeric_vector(value, function, P::InputSize.canonical_name())?;
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
    use runmat_value::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericStorage, Tensor,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn feature_input_metadata_preserves_low_rank_axis_admission() {
        for (dimensions, admitted) in [(Vec::new(), false), (vec![2], true)] {
            let count = dimensions.iter().product();
            let input =
                Value::Tensor(Tensor::from_f32(vec![1.0; count], dimensions.clone()).unwrap());
            let (shape, precision, format, payload_kind) = metadata(&input, "forward").unwrap();
            assert_eq!(shape, dimensions.as_slice());
            assert_eq!(precision, NumericDType::F32);
            assert_eq!(payload_kind, InputPayloadKind::Real);
            let axis_plan = axes(
                &format,
                DeepLearningInputKind::Feature,
                shape,
                ChannelPolicy::Required,
                "forward",
            );
            assert_eq!(axis_plan.is_ok(), admitted);
            if let Ok(axis_plan) = axis_plan {
                assert_eq!(axis_plan.shape(), dimensions.as_slice());
                assert_eq!(axis_plan.layout().axes(), &[A::Channel]);
            }
        }
    }

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
            let (shape, precision, format, payload_kind) = metadata(&input, "forward").unwrap();
            assert_eq!(shape, &[1, 2]);
            assert_eq!(precision, NumericDType::F64);
            assert!(format.is_empty());
            assert_eq!(payload_kind, InputPayloadKind::Real);
            let Value::Tensor(tensor) = input else {
                unreachable!()
            };
            assert_eq!(tensor.numeric_dtype(), original_dtype);
        }
    }

    #[test]
    fn integer_complex_input_is_rejected_before_any_host_projection() {
        let components = IntegerComplexStorage::new(
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I32(vec![3, 4]),
        )
        .unwrap();
        let input =
            Value::ComplexTensor(ComplexTensor::new_integer(components, vec![1, 2]).unwrap());
        assert!(metadata(&input, "forward").is_err());
    }
}
