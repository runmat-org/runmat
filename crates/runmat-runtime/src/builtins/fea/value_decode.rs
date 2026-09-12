use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::MODEL_NAME;
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::options_json::normalize_token;
use crate::builtins::fea::study::ModelDefaultsMode;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::fea) fn logical_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected logical scalar; got {other:?}"),
        )),
    }
}

pub(in crate::builtins::fea) fn exact_bool_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<bool> {
    if let Ok(value) = logical_from_value(builtin, value) {
        return Ok(value);
    }
    if let Some(integer) = tensor_utils::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "numeric logical option must be exactly zero or one",
            )),
        };
    }
    match ordinary_double_scalar(value) {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "logical option must be a logical scalar or exact numeric zero or one",
        )),
    }
}

pub(in crate::builtins::fea) fn bool_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<bool> {
    bool::try_from(value).map_err(|err| builtin_error(builtin, &ERROR_INPUT, err))
}

pub(in crate::builtins::fea) fn usize_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<usize> {
    if let Some(int) = tensor_utils::scalar_integer_value(value) {
        return int.try_to_usize().ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "expected non-negative integer value outside the platform range",
            )
        });
    }
    match ordinary_double_scalar(value) {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => {
            if n > usize::MAX as f64 || (usize::BITS == 64 && n == usize::MAX as f64) {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "expected non-negative integer value outside the platform range",
                ));
            }
            Ok(n as usize)
        }
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected non-negative integer value; got {value:?}"),
        )),
    }
}

pub(in crate::builtins::fea) fn ordinary_double_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == runmat_value::NumericDType::F64 =>
        {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

pub(in crate::builtins::fea) fn string_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| scalar_string(item, builtin, &ERROR_INPUT))
            .collect(),
        Value::StringArray(array) => Ok(array.data.clone()),
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

pub(in crate::builtins::fea) fn usize_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor
                .shape
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

pub(in crate::builtins::fea) fn one_based_usize_vec_from_value(
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

pub(in crate::builtins::fea) fn parse_model_defaults_mode(
    text: &str,
) -> BuiltinResult<ModelDefaultsMode> {
    match normalize_token(text).as_str() {
        "profilescaffold" | "scaffold" | "profile" => Ok(ModelDefaultsMode::ProfileScaffold),
        "none" | "empty" => Ok(ModelDefaultsMode::None),
        other => Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            format!("unsupported model defaults mode `{other}`"),
        )),
    }
}
