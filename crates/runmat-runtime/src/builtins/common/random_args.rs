use crate::builtins::common::tensor;
use runmat_value::{ComplexTensor, IntValue, NumericScalar, Value};

#[cfg(test)]
mod owned_constructor_tests;

pub(crate) fn validate_constructor_gpu_output(
    label: &str,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
) -> crate::BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    validate_constructor_gpu_output_protecting(
        label,
        provider,
        output,
        expected_shape,
        expected_storage,
        expected_precision,
        expected_integer,
        expected_logical,
        &[],
    )
}

/// Validate a newly allocated constructor result without adopting, relabelling,
/// or releasing borrowed inputs (including a `like` prototype).
pub(crate) fn validate_constructor_gpu_output_protecting(
    label: &str,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    validate_constructor_gpu_storage_protecting(
        label,
        provider,
        output,
        expected_shape,
        expected_storage,
        expected_precision,
        expected_integer,
        expected_logical,
        protected,
    )?
    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
    .map_err(|source| {
        crate::build_runtime_error(format!(
            "{label}: cannot adopt constructor result: {source}"
        ))
        .with_builtin(label)
        .with_source(source)
        .build()
    })
}

/// Validate physical ownership/class without allocating a replacement view.
/// Intermediate provider outputs use this until their final language result
/// is known, so each retry does not mint another retained view claim.
pub(crate) fn validate_constructor_gpu_storage_protecting(
    label: &str,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<runmat_accelerate_api::GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{exact_provider_for_handle, reject_gpu_output};
    let expected_class = crate::builtins::common::gpu_helpers::expected_gpu_class_identity(
        expected_precision,
        expected_integer,
        expected_logical,
    );
    let expected_element = expected_integer
        .map(runmat_accelerate_api::NumericElementType::from)
        .or_else(|| {
            expected_precision.map(|precision| match precision {
                runmat_accelerate_api::ProviderPrecision::F32 => {
                    runmat_accelerate_api::NumericElementType::F32
                }
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    runmat_accelerate_api::NumericElementType::F64
                }
            })
        });
    let existing_class = runmat_accelerate_api::handle_class_identity(&output);
    let valid = expected_class.is_some()
        && !protected
            .iter()
            .any(|input| output.shares_allocation_with(input))
        && output.device_id() == provider.device_id()
        && output.shape() == expected_shape
        && exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && output.descriptor().storage == Some(expected_storage)
        && (if expected_logical {
            expected_storage == runmat_accelerate_api::GpuTensorStorage::Real
                && matches!(
                    output.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::U8
                        | runmat_accelerate_api::NumericElementType::F32
                        | runmat_accelerate_api::NumericElementType::F64
                )
        } else {
            output.descriptor().element_type == expected_element
        })
        && runmat_accelerate_api::handle_is_logical(&output) == expected_logical
        && existing_class
            .as_ref()
            .is_some_and(|class_identity| Some(class_identity) == expected_class.as_ref());
    if !valid {
        return Err(reject_gpu_output(
            output,
            crate::build_runtime_error(format!(
                "{label}: provider returned an invalid constructor result"
            ))
            .with_builtin(label)
            .build(),
        ));
    }
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConstructorDimensions {
    pub values: Vec<usize>,
    pub is_column_vector: bool,
}

/// Parse MATLAB array-constructor dimensions without passing exact integer
/// storage through the floating compatibility representation. Signed negative
/// sizes clamp to zero, as documented by the core array constructors.
#[async_recursion::async_recursion(?Send)]
pub(crate) async fn extract_constructor_dimensions(
    value: &Value,
    label: &str,
) -> Result<Option<ConstructorDimensions>, String> {
    if matches!(value, Value::LogicalArray(_) | Value::Bool(_)) {
        return Ok(None);
    }
    match value {
        Value::Num(value) => Ok(Some(ConstructorDimensions {
            values: vec![parse_constructor_float_dimension(*value, label)?],
            is_column_vector: false,
        })),
        Value::Int(value) => Ok(Some(ConstructorDimensions {
            values: vec![parse_constructor_integer_dimension(value, label)?],
            is_column_vector: false,
        })),
        Value::Tensor(tensor) => {
            let len = tensor.len();
            if len == 0 {
                return Ok(Some(ConstructorDimensions {
                    values: Vec::new(),
                    is_column_vector: false,
                }));
            }
            let scalar = len == 1;
            let shape = tensor.shape();
            let row = shape.len() >= 2 && shape[0] == 1;
            let column = shape.len() >= 2 && shape[1] == 1;
            if !(scalar || row || column || shape.len() == 1) {
                return Ok(None);
            }
            let values = (0..len)
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .ok_or_else(|| format!("{label}: missing dimension value"))
                        .and_then(|value| parse_constructor_numeric_dimension(value, label))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(ConstructorDimensions {
                values,
                is_column_vector: column && !row && len > 1,
            }))
        }
        Value::GpuTensor(_) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|error| format!("{label}: {error}"))?;
            extract_constructor_dimensions(&gathered, label).await
        }
        _ => Ok(None),
    }
}

pub(crate) fn normalize_constructor_shape(mut dimensions: Vec<usize>) -> Vec<usize> {
    let mut shape = match dimensions.len() {
        0 => vec![0, 0],
        1 => vec![dimensions[0], dimensions[0]],
        _ => {
            while dimensions.len() > 2 && dimensions.last() == Some(&1) {
                dimensions.pop();
            }
            dimensions
        }
    };
    if shape.len() == 1 {
        shape.push(1);
    }
    shape
}

fn parse_constructor_numeric_dimension(value: NumericScalar, label: &str) -> Result<usize, String> {
    checked_constructor_numeric_dimension(value).map_err(|error| error.message(label))
}

fn parse_constructor_float_dimension(value: f64, label: &str) -> Result<usize, String> {
    checked_constructor_float_dimension(value).map_err(|error| error.message(label))
}

fn parse_constructor_integer_dimension(value: &IntValue, label: &str) -> Result<usize, String> {
    checked_constructor_integer_dimension(value).map_err(|error| error.message(label))
}

/// The selected-source path can inspect the same dimension policy without
/// formatting an error before its host metadata admission. The ordinary
/// constructor remains the owner of the source-level diagnostic text.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ConstructorDimensionFailure {
    NonIntegral,
    OutOfRange,
}

impl ConstructorDimensionFailure {
    fn message(self, label: &str) -> String {
        match self {
            Self::NonIntegral => format!("{label}: dimensions must be finite integer values"),
            Self::OutOfRange => {
                format!("{label}: dimension is outside the supported platform range")
            }
        }
    }
}

pub(crate) fn checked_constructor_numeric_dimension(
    value: NumericScalar,
) -> Result<usize, ConstructorDimensionFailure> {
    match value {
        NumericScalar::F64(value) => checked_constructor_float_dimension(value),
        NumericScalar::F32(value) => checked_constructor_float_dimension(f64::from(value)),
        integer => checked_constructor_integer_dimension(
            &integer
                .into_int_value()
                .expect("nonfloating numeric scalar is an integer"),
        ),
    }
}

pub(crate) fn checked_constructor_float_dimension(
    value: f64,
) -> Result<usize, ConstructorDimensionFailure> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(ConstructorDimensionFailure::NonIntegral);
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(ConstructorDimensionFailure::OutOfRange);
    }
    Ok(value as usize)
}

pub(crate) fn checked_constructor_integer_dimension(
    value: &IntValue,
) -> Result<usize, ConstructorDimensionFailure> {
    let parsed = match value {
        IntValue::I8(value) => usize::try_from((*value).max(0)),
        IntValue::I16(value) => usize::try_from((*value).max(0)),
        IntValue::I32(value) => usize::try_from((*value).max(0)),
        IntValue::I64(value) => usize::try_from((*value).max(0)),
        IntValue::U8(value) => Ok(usize::from(*value)),
        IntValue::U16(value) => Ok(usize::from(*value)),
        IntValue::U32(value) => usize::try_from(*value),
        IntValue::U64(value) => usize::try_from(*value),
    };
    parsed.map_err(|_| ConstructorDimensionFailure::OutOfRange)
}

/// Extract a lowercased keyword from runtime values such as strings or
/// single-row char arrays.
pub(crate) fn keyword_of(value: &Value) -> crate::BuiltinResult<Option<String>> {
    crate::builtins::common::arg_tokens::scalar_keyword_of(value, "arrayCreation")
}

/// Attempt to parse a dimension argument. Returns `Ok(Some(Vec))` when the
/// value encodes dimensions, `Ok(None)` when the value is not a dimension
/// argument, and `Err` when the value is dimension-like but invalid.
pub(crate) async fn extract_dims(value: &Value, label: &str) -> Result<Option<Vec<usize>>, String> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(handle.shape()) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(format!("{label}: {err}"))
            }
        }
    }
}

/// Determine the output shape encoded by a runtime value.
pub(crate) fn shape_from_value(value: &Value, label: &str) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape().to_vec()),
        Value::ComplexTensor(t) => Ok(t.shape().to_vec()),
        Value::LogicalArray(l) => Ok(l.shape.to_vec()),
        Value::GpuTensor(h) => Ok(h.shape().to_vec()),
        Value::CharArray(ca) => Ok(ca.shape().to_vec()),
        Value::Cell(cell) => Ok(cell.shape().to_vec()),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::StringArray(_) => Ok(vec![1, 1]),
        other => Err(format!("{label}: unsupported prototype {other:?}")),
    }
}

/// Convert a complex tensor back into an appropriate runtime value.
pub(crate) fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == runmat_value::NumericDType::F64 {
        let (re, im) = tensor
            .as_f64_slice()
            .expect("double complex tensor")
            .first()
            .copied()
            .expect("scalar complex tensor")
            .into();
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
mod constructor_dimension_tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn constructor_validation_releases_rejected_output_through_its_actual_owner() {
        use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, ProviderPrecision};
        crate::builtins::common::test_support::with_rejecting_native_result_provider(|provider| {
            // This fixture deliberately returns an allocation owned by another provider.
            let dummy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let output = provider
                .cross(&dummy, &dummy, None)
                .expect("foreign-owner result");
            assert_ne!(output.device_id(), provider.device_id());
            let retained = output.clone();
            let physical = output.allocation_lease().downgrade();
            let before = provider.free_count();
            let error = validate_constructor_gpu_output(
                "constructor",
                provider,
                output,
                &[1, 1],
                GpuTensorStorage::ComplexInterleaved,
                Some(ProviderPrecision::F64),
                None,
                false,
            )
            .expect_err("wrong owner must not be adopted");
            assert!(error.contains("invalid constructor result"));
            assert_eq!(provider.free_count(), before);
            retained.allocation_lease().validate_live().unwrap();
            drop(retained);
            assert!(physical.acquire().is_none());
            assert_eq!(provider.free_count(), before + 1);
            assert!(block_on(provider.download(&dummy)).is_ok());
        });
    }

    #[test]
    fn constructor_validation_preserves_borrowed_prototype_and_its_metadata() {
        use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let prototype = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("prototype");
            let original_class = runmat_accelerate_api::handle_class_identity(&prototype);
            let original_logical = runmat_accelerate_api::handle_is_logical(&prototype);
            for (expected_shape, expected_logical) in [
                (&[1, 2][..], false),
                (&[1, 2][..], true),
                (&[2, 1][..], false),
            ] {
                let error = validate_constructor_gpu_output_protecting(
                    "constructor",
                    provider,
                    prototype.clone(),
                    expected_shape,
                    GpuTensorStorage::Real,
                    Some(ProviderPrecision::F64),
                    None,
                    expected_logical,
                    &[&prototype],
                )
                .expect_err("borrowed alias must not be adopted or freed");
                assert!(error.contains("invalid constructor result"));
                assert_eq!(
                    runmat_accelerate_api::handle_class_identity(&prototype),
                    original_class
                );
                assert_eq!(
                    runmat_accelerate_api::handle_is_logical(&prototype),
                    original_logical
                );
                assert_eq!(
                    block_on(provider.download(&prototype))
                        .expect("prototype remains live")
                        .data,
                    vec![1.0, 2.0]
                );
            }
            drop(prototype);
        });
    }

    #[test]
    fn exact_integer_size_vectors_cover_every_integer_class() {
        let storages = [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("size vector");
            let parsed = block_on(extract_constructor_dimensions(
                &Value::Tensor(tensor),
                "constructor",
            ))
            .expect("parse")
            .expect("dimensions");
            assert_eq!(parsed.values, vec![2, 3]);
            assert!(!parsed.is_column_vector);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn constructor_dimensions_preserve_scalar_and_rank_one_shapes() {
        for (shape, values) in [(vec![], vec![3.0]), (vec![2], vec![2.0, 3.0])] {
            let value = Value::Tensor(Tensor::new(values.clone(), shape.clone()).unwrap());
            let parsed = block_on(extract_constructor_dimensions(&value, "constructor"))
                .unwrap()
                .unwrap();
            assert_eq!(
                parsed.values,
                values.iter().map(|v| *v as usize).collect::<Vec<_>>()
            );
            assert!(!parsed.is_column_vector);
            assert_eq!(shape_from_value(&value, "constructor").unwrap(), shape);
        }
    }

    #[test]
    fn signed_negative_sizes_clamp_and_column_vectors_are_identified() {
        let negative = Value::Int(IntValue::I64(-7));
        let parsed = block_on(extract_constructor_dimensions(&negative, "constructor"))
            .expect("parse")
            .expect("dimensions");
        assert_eq!(parsed.values, vec![0]);

        let column = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1])
            .expect("column size vector");
        let parsed = block_on(extract_constructor_dimensions(
            &Value::Tensor(column),
            "constructor",
        ))
        .expect("parse")
        .expect("dimensions");
        assert!(parsed.is_column_vector);
    }

    #[test]
    fn constructor_shapes_drop_only_trailing_singletons_beyond_dimension_two() {
        assert_eq!(normalize_constructor_shape(vec![3]), vec![3, 3]);
        assert_eq!(normalize_constructor_shape(vec![3, 1, 1, 1]), vec![3, 1]);
        assert_eq!(normalize_constructor_shape(vec![3, 1, 2, 1]), vec![3, 1, 2]);
        assert_eq!(normalize_constructor_shape(Vec::new()), vec![0, 0]);
    }

    #[test]
    fn constructor_validation_never_relabels_physical_double_as_single() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload");
            let error = validate_constructor_gpu_output(
                "constructor",
                provider,
                handle,
                &[1, 2],
                runmat_accelerate_api::GpuTensorStorage::Real,
                Some(runmat_accelerate_api::ProviderPrecision::F32),
                None,
                false,
            )
            .expect_err("physical double output cannot satisfy single precision");
            assert!(error.contains("invalid constructor result"));
        });
    }

    #[test]
    fn constructor_validation_ignores_stale_floating_side_metadata_on_integer_output() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let values = [1_i32, 2_i32];
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I32(&values),
                    shape: &[1, 2],
                })
                .expect("integer upload");
            let output = validate_constructor_gpu_output(
                "constructor",
                provider,
                handle,
                &[1, 2],
                runmat_accelerate_api::GpuTensorStorage::Real,
                None,
                Some(runmat_accelerate_api::IntegerElementType::I32),
                false,
            )
            .expect("durable integer descriptor must override stale side metadata");
            assert_eq!(
                output.descriptor().element_type,
                Some(runmat_accelerate_api::NumericElementType::I32)
            );
        });
    }
}
