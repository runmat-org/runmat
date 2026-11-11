use super::{builtin_error, scalar_string, tensor_utils, usize_from_value, ERROR_INPUT};
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) fn string_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|item| scalar_string(item, builtin, &ERROR_INPUT))
            .collect(),
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| scalar_string(&Value::String(text.clone()), builtin, &ERROR_INPUT))
            .collect(),
        Value::String(_) | Value::CharArray(_) => {
            Ok(vec![scalar_string(value, builtin, &ERROR_INPUT)?])
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected string, string array, or cell array of strings; got {other:?}"),
        )),
    }
}

pub(super) fn usize_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor
                .shape()
                .iter()
                .filter(|&&dimension| dimension > 1)
                .count()
                > 1
            {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "expected a numeric scalar or vector of indices, not a matrix",
                ));
            }
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .into_iter()
                    .map(|value| usize_from_value(builtin, &Value::Int(value)))
                    .collect();
            }
            if tensor.numeric_dtype() != runmat_value::NumericDType::F64 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "floating index selectors must use ordinary double storage",
                ));
            }
            tensor_utils::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| usize_from_value(builtin, &Value::Num(value)))
                .collect()
        }
        Value::Int(_) | Value::Num(_) => Ok(vec![usize_from_value(builtin, value)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected numeric scalar or vector of indices; got {other:?}"),
        )),
    }
}

pub(super) fn one_based_usize_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    usize_vec_from_value(builtin, value)?
        .into_iter()
        .map(|value| {
            value.checked_sub(1).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "result indices are one-based and must be positive",
                )
            })
        })
        .collect()
}
