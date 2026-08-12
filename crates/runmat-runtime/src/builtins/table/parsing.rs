use super::*;

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(invalid_argument(format!(
            "table: {context} must be a string scalar or character vector"
        ))),
    }
}

pub(super) fn bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(value) => Ok(!value.is_zero()),
        Value::Num(value) if value.is_finite() => Ok(*value != 0.0),
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let text = scalar_text(value, context)?;
            match text.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" => Ok(true),
                "false" | "off" | "no" => Ok(false),
                _ => Err(invalid_argument(format!(
                    "table: {context} must be logical"
                ))),
            }
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be logical"
        ))),
    }
}

pub(super) fn zero_one_bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.try_to_u64() == Some(1) => Ok(true),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0) {
            Some(runmat_value::NumericScalar::F64(0.0)) => Ok(false),
            Some(runmat_value::NumericScalar::F64(1.0)) => Ok(true),
            Some(runmat_value::NumericScalar::F32(0.0)) => Ok(false),
            Some(runmat_value::NumericScalar::F32(1.0)) => Ok(true),
            Some(value) if value.into_int_value().is_some_and(|value| value.is_zero()) => Ok(false),
            Some(value)
                if value
                    .into_int_value()
                    .is_some_and(|value| value.try_to_u64() == Some(1)) =>
            {
                Ok(true)
            }
            _ => Err(invalid_argument(format!(
                "table: {context} must be scalar logical or numeric 0 or 1"
            ))),
        },
        _ => Err(invalid_argument(format!(
            "table: {context} must be scalar logical or numeric 0 or 1"
        ))),
    }
}

pub(super) fn nonnegative_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => value.try_to_usize().ok_or_else(|| {
            invalid_argument(format!("table: {context} must be a non-negative integer"))
        }),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value.try_to_usize().ok_or_else(|| {
                    invalid_argument(format!("table: {context} must be a non-negative integer"))
                });
            }
            nonnegative_usize_from_f64(
                crate::builtins::common::tensor::tensor_value_f64(tensor, 0),
                context,
            )
        }
        Value::Num(value)
            if value.is_finite()
                && *value >= 0.0
                && (value.round() - value).abs() <= f64::EPSILON =>
        {
            nonnegative_usize_from_f64(value.round(), context)
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be a non-negative integer"
        ))),
    }
}

fn nonnegative_usize_from_f64(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(invalid_argument(format!(
            "table: {context} must be a non-negative integer"
        )));
    }
    Ok(value as usize)
}

pub(super) fn positive_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    let value = nonnegative_usize(value, context)?;
    if value == 0 {
        return Err(invalid_argument(format!(
            "table: {context} must be a positive integer"
        )));
    }
    Ok(value)
}

pub(super) fn option_value_is_empty(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(array) => {
            array.data.is_empty()
                || (array.rows == 1 && array.data.iter().all(|ch| ch.is_whitespace()))
        }
        Value::StringArray(array) => {
            array.data.is_empty() || (array.data.len() == 1 && array.data[0].trim().is_empty())
        }
        Value::Tensor(tensor) => crate::builtins::common::tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => {
            cell.data.is_empty() || cell.data.iter().all(|handle| option_value_is_empty(handle))
        }
        _ => false,
    }
}

pub(super) fn string_list(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(ca) if ca.rows == 1 => Ok(vec![ca.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                let value = handle;
                out.extend(string_list(value)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument(
            "table: expected string, string array, character vector, or cellstr",
        )),
    }
}

pub(super) fn optional_raw_variable_name_list(value: &Value) -> BuiltinResult<Option<Vec<String>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        raw_variable_name_list(value).map(Some)
    }
}

pub(super) fn raw_variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let names = string_list(value)?;
    if names.is_empty() {
        return Err(invalid_variable("table: variable names must not be empty"));
    }
    Ok(names)
}

pub(super) fn variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let names = raw_variable_name_list(value)?;
    validate_variable_names(&names)?;
    Ok(names)
}

pub(super) fn optional_variable_type_list(
    value: &Value,
) -> BuiltinResult<Option<Vec<ImportVariableType>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        variable_type_list(value).map(Some)
    }
}

pub(super) fn variable_type_list(value: &Value) -> BuiltinResult<Vec<ImportVariableType>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::parse(raw))
        .collect()
}

pub(super) fn variable_type_names(value: &Value) -> BuiltinResult<Vec<String>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::canonical_label(raw))
        .collect()
}

pub(super) fn optional_sheet_selector(value: &Value) -> BuiltinResult<Option<SheetSelector>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        SheetSelector::parse(value).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    fn integer_storages(values: &[u64]) -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(values.iter().map(|&value| value as i8).collect()),
            IntegerStorage::I16(values.iter().map(|&value| value as i16).collect()),
            IntegerStorage::I32(values.iter().map(|&value| value as i32).collect()),
            IntegerStorage::I64(values.iter().map(|&value| value as i64).collect()),
            IntegerStorage::U8(values.iter().map(|&value| value as u8).collect()),
            IntegerStorage::U16(values.iter().map(|&value| value as u16).collect()),
            IntegerStorage::U32(values.iter().map(|&value| value as u32).collect()),
            IntegerStorage::U64(values.to_vec()),
        ]
    }

    #[test]
    fn table_usize_parsers_read_typed_integer_storage_exactly() {
        let exact = (1_u64 << 53) + 1;
        let count =
            Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).expect("count");

        let parsed = nonnegative_usize(&Value::Tensor(count), "head row count");
        if usize::BITS == 64 {
            assert_eq!(parsed.unwrap(), exact as usize);
        } else {
            assert!(parsed.is_err());
        }

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("count");
        assert!(nonnegative_usize(&Value::Tensor(negative), "head row count").is_err());
    }

    #[test]
    fn table_usize_parsers_ignore_poisoned_mirrors_for_every_integer_class() {
        for storage in integer_storages(&[2]) {
            let count = Tensor::new_integer(storage, vec![1, 1]).expect("count");
            assert_eq!(
                nonnegative_usize(&Value::Tensor(count), "count").unwrap(),
                2
            );
        }
    }

    #[test]
    fn option_empty_check_uses_typed_integer_storage_length() {
        let scalar = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        assert!(!option_value_is_empty(&Value::Tensor(scalar)));

        let empty = Tensor::new_integer(IntegerStorage::U8(Vec::new()), vec![0, 0]).unwrap();
        assert!(option_value_is_empty(&Value::Tensor(empty)));
    }

    #[test]
    fn table_usize_parsers_reject_unrepresentable_double_bounds() {
        let boundary = nonnegative_usize(&Value::Num(usize::MAX as f64), "head row count");
        if usize::BITS == 64 {
            assert!(boundary.is_err());
        } else {
            assert_eq!(boundary.unwrap(), usize::MAX);
        }
        assert!(
            nonnegative_usize(&Value::Num((usize::MAX as f64) + 1.0), "head row count").is_err()
        );
        assert!(nonnegative_usize(&Value::Num(1.5), "head row count").is_err());
        assert!(positive_usize(&Value::Num(0.0), "head row count").is_err());
    }
}
