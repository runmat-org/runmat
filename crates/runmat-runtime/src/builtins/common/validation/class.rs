//! Class and underlying-type predicates.

use super::{
    class_name_for_value, handle_integer_type, handle_is_logical, tensor, underlying_type_matches,
    value_has_logical_class, value_has_native_integer_class, CharArray, ComplexTensor,
    NumericDType, RuntimeError, SparseTensor, Value,
};

pub fn value_is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

pub fn value_is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Complex(_, _) => true,
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none(),
        Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F64 | NumericDType::F32),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            !handle_is_logical(handle) && handle_integer_type(handle).is_none()
        }
        _ => false,
    }
}

pub fn value_is_numeric_or_logical(value: &Value) -> bool {
    value_is_numeric(value) || value_has_logical_class(value)
}

pub fn value_matches_class(value: &Value, class_name: &str) -> bool {
    let requested = class_name.trim();
    if requested.is_empty() {
        return false;
    }
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return requested.eq_ignore_ascii_case("gpuarray");
    }
    match requested.to_ascii_lowercase().as_str() {
        "numeric" => value_is_numeric(value),
        "float" => value_is_float(value),
        "integer" => value_has_native_integer_class(value),
        "logical" => value_has_logical_class(value),
        "char" => matches!(value, Value::CharArray(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_) | Value::StructArray(_)),
        "sparse" => matches!(value, Value::SparseTensor(_)),
        "double" => {
            matches!(value, Value::Num(_) | Value::Complex(_, _))
                || matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F64)
                || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_none())
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F64)
                || matches!(value, Value::GpuTensor(handle) if !handle_is_logical(handle) && handle_integer_type(handle).is_none() && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F64))
        }
        "single" => {
            matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle) if !handle_is_logical(handle) && handle_integer_type(handle).is_none() && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
        }
        "gpuarray" => {
            matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        }
        _ => class_name_for_value(value).eq_ignore_ascii_case(requested),
    }
}

pub fn must_be_a(value: &Value, class_names: Vec<String>) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| value_matches_class(value, class_name)))
}

pub fn value_underlying_type_matches(
    value: &Value,
    class_names: Vec<String>,
) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| underlying_type_matches(value, class_name)))
}
