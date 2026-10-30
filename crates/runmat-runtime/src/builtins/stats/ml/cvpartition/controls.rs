use super::*;
use std::borrow::Cow;

pub(super) fn materialized_mask_len(rows: usize, cols: usize, label: &str) -> BuiltinResult<usize> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_error(format!("cvpartition: {label} size overflow")))?;
    if len > MAX_MATERIALIZED_MASK_CELLS {
        return Err(internal_error(format!(
            "cvpartition: {label} would materialize {len} logical entries; limit is {MAX_MATERIALIZED_MASK_CELLS}"
        )));
    }
    Ok(len)
}

pub(super) fn holdout_count(value: &Value, n: usize) -> BuiltinResult<usize> {
    if n < 2 {
        return Err(invalid_argument(
            "cvpartition: Holdout requires at least two observations",
        ));
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|count| *count > 0 && *count < n)
            .ok_or_else(|| {
                invalid_argument(
                    "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
                )
            });
    }
    let raw = scalar_number(value, "Holdout")?;
    if raw > 0.0 && raw < 1.0 {
        let count = (raw * n as f64).round() as usize;
        return Ok(count.clamp(1, n.saturating_sub(1)));
    }
    if raw.fract() == 0.0 && raw >= 1.0 && raw < n as f64 {
        return Ok(raw as usize);
    }
    Err(invalid_argument(
        "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

pub(super) fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: {label} must be a positive integer scalar"
                ))
            });
    }
    positive_integer_number(scalar_number(value, label)?, label)
}

pub(super) fn positive_integer_number(raw: f64, label: &str) -> BuiltinResult<usize> {
    PartitionCountGrammar::CvPartition
        .positive_from_float(raw)
        .ok_or_else(|| {
            invalid_argument(format!(
                "cvpartition: {label} must be a positive integer scalar"
            ))
        })
}

pub(super) fn scalar_number(value: &Value, label: &str) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(number) => *number,
        Value::Int(integer) => integer.to_f64(),
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid_argument(format!(
            "cvpartition: {label} must be finite"
        )));
    }
    Ok(number)
}

pub(super) fn scalar_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(number) => parse_stratify_numeric_boolean(*number).ok_or_else(|| {
            invalid_argument(format!("cvpartition: {label} must be logical scalar"))
        }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                match integer.try_to_usize() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(invalid_argument(format!(
                        "cvpartition: {label} must be logical scalar"
                    ))),
                }
            } else {
                parse_stratify_numeric_boolean(tensor::tensor_value_f64(tensor, 0)).ok_or_else(
                    || invalid_argument(format!("cvpartition: {label} must be logical scalar")),
                )
            }
        }
        Value::Int(integer) => match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid_argument(format!(
                "cvpartition: {label} must be logical scalar"
            ))),
        },
        Value::String(runmat_value::StringElement::Text(text)) => {
            parse_stratify_boolean_alias(text).ok_or_else(|| {
                invalid_argument(format!("cvpartition: {label} must be logical scalar"))
            })
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text = runmat_value::decode_utf16_code_units(chars.data()).map_err(|error| {
                invalid_argument(format!("cvpartition: {label} contains {error}"))
            })?;
            parse_stratify_boolean_alias(&text).ok_or_else(|| {
                invalid_argument(format!("cvpartition: {label} must be logical scalar"))
            })
        }
        _ => Err(invalid_argument(format!(
            "cvpartition: {label} must be logical scalar"
        ))),
    }
}

pub(super) fn scalar_text<'a>(value: &'a Value, label: &str) -> BuiltinResult<Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Cow::Borrowed(text.as_str())),
        Value::CharArray(chars) if chars.rows <= 1 => {
            runmat_value::decode_utf16_code_units(chars.data())
                .map(Cow::Owned)
                .map_err(|error| invalid_argument(format!("cvpartition: {label} contains {error}")))
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Cow::Borrowed(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        other => Err(invalid_argument(format!(
            "cvpartition: {label} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn value_partition_method(value: &Value) -> Option<PartitionKind> {
    scalar_text(value, "partition kind")
        .ok()
        .and_then(|text| PartitionKind::parse(&text))
}

pub(super) fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}
