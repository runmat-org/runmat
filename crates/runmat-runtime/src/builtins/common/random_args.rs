use crate::builtins::common::tensor;
use runmat_value::{ComplexTensor, IntValue, NumericScalar, Value};

pub(crate) fn validate_constructor_gpu_output(
    label: &str,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
) -> Result<runmat_accelerate_api::GpuTensorHandle, String> {
    let expected_class = if expected_logical {
        "logical"
    } else if let Some(integer) = expected_integer {
        match integer {
            runmat_accelerate_api::IntegerElementType::I8 => "int8",
            runmat_accelerate_api::IntegerElementType::I16 => "int16",
            runmat_accelerate_api::IntegerElementType::I32 => "int32",
            runmat_accelerate_api::IntegerElementType::I64 => "int64",
            runmat_accelerate_api::IntegerElementType::U8 => "uint8",
            runmat_accelerate_api::IntegerElementType::U16 => "uint16",
            runmat_accelerate_api::IntegerElementType::U32 => "uint32",
            runmat_accelerate_api::IntegerElementType::U64 => "uint64",
        }
    } else {
        match expected_precision {
            Some(runmat_accelerate_api::ProviderPrecision::F32) => "single",
            Some(runmat_accelerate_api::ProviderPrecision::F64) => "double",
            None => "",
        }
    };
    let existing_precision = runmat_accelerate_api::handle_precision(&output);
    let effective_precision = expected_precision
        .is_some()
        .then(|| existing_precision.unwrap_or_else(|| provider.precision()));
    let existing_class = runmat_accelerate_api::handle_class_name(&output);
    let valid = output.device_id == provider.device_id()
        && output.shape == expected_shape
        && runmat_accelerate_api::provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == expected_storage
        && effective_precision == expected_precision
        && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
        && (expected_integer.is_none() || existing_precision.is_none())
        && (!runmat_accelerate_api::handle_is_logical(&output) || expected_logical)
        && existing_class
            .as_deref()
            .is_none_or(|class_name| class_name == expected_class);
    if !valid {
        let _ = provider.free(&output);
        return Err(format!(
            "{label}: provider returned an invalid constructor result"
        ));
    }
    match expected_precision {
        Some(precision) => runmat_accelerate_api::set_handle_precision(&output, precision),
        None => runmat_accelerate_api::clear_handle_precision(&output),
    }
    runmat_accelerate_api::set_handle_logical(&output, expected_logical);
    runmat_accelerate_api::set_handle_class_name(&output, expected_class);
    runmat_accelerate_api::mark_handle_explicit(&output);
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
            let row = tensor.shape.len() >= 2 && tensor.shape[0] == 1;
            let column = tensor.shape.len() >= 2 && tensor.shape[1] == 1;
            if !(scalar || row || column || tensor.shape.len() == 1) {
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
    match value {
        NumericScalar::F64(value) => parse_constructor_float_dimension(value, label),
        NumericScalar::F32(value) => parse_constructor_float_dimension(f64::from(value), label),
        integer => parse_constructor_integer_dimension(
            &integer
                .into_int_value()
                .expect("nonfloating numeric scalar is an integer"),
            label,
        ),
    }
}

fn parse_constructor_float_dimension(value: f64, label: &str) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(format!("{label}: dimensions must be finite integer values"));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(format!(
            "{label}: dimension is outside the supported platform range"
        ));
    }
    Ok(value as usize)
}

fn parse_constructor_integer_dimension(value: &IntValue, label: &str) -> Result<usize, String> {
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
    parsed.map_err(|_| format!("{label}: dimension is outside the supported platform range"))
}

/// Extract a lowercased keyword from runtime values such as strings or
/// single-row char arrays.
pub(crate) fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

/// Attempt to parse a dimension argument. Returns `Ok(Some(Vec))` when the
/// value encodes dimensions, `Ok(None)` when the value is not a dimension
/// argument, and `Err` when the value is dimension-like but invalid.
pub(crate) async fn extract_dims(value: &Value, label: &str) -> Result<Option<Vec<usize>>, String> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
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
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(ca.shape.clone()),
        Value::Cell(cell) => Ok(cell.shape.clone()),
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
            .expect("scalar complex tensor");
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
    fn constructor_validation_rejects_integer_output_with_floating_precision_metadata() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let values = [1_i32, 2_i32];
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I32(&values),
                    shape: &[1, 2],
                })
                .expect("integer upload");
            runmat_accelerate_api::set_handle_precision(
                &handle,
                runmat_accelerate_api::ProviderPrecision::F64,
            );
            let error = validate_constructor_gpu_output(
                "constructor",
                provider,
                handle,
                &[1, 2],
                runmat_accelerate_api::GpuTensorStorage::Real,
                None,
                Some(runmat_accelerate_api::IntegerElementType::I32),
                false,
            )
            .expect_err("integer output cannot carry floating precision metadata");
            assert!(error.contains("invalid constructor result"));
        });
    }
}
