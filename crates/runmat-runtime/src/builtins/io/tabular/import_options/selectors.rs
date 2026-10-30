use super::invalid;
use crate::BuiltinResult;
use runmat_value::{IntValue, Value};

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| invalid("import options: expected a string scalar or character row"))
}

pub(super) fn text_list(value: &Value) -> BuiltinResult<Vec<String>> {
    if let Ok(text) = scalar_text(value) {
        return Ok(vec![text]);
    }
    match value {
        Value::StringArray(values) if values.data().iter().all(|text| !text.is_missing()) => {
            Ok(values
                .data()
                .iter()
                .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
                .collect())
        }
        Value::Cell(values) => values.data().iter().map(scalar_text).collect(),
        _ => Err(invalid(
            "import options: expected a text scalar, string array, or cellstr",
        )),
    }
}

pub(super) fn resolve(value: &Value, names: &[String]) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(number) => Ok(vec![one_based_float(*number, names.len())?]),
        Value::Int(integer) => Ok(vec![one_based_integer(integer, names.len())?]),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(|integer| one_based_integer(integer, names.len()))
                    .collect();
            }
            (0..tensor.len())
                .map(|index| {
                    one_based_float(
                        tensor
                            .numeric_value_at(index)
                            .expect("numeric tensor lane")
                            .materialize_f64(),
                        names.len(),
                    )
                })
                .collect()
        }
        Value::LogicalArray(mask) => {
            if mask.data.len() != names.len() {
                return Err(invalid(
                    "import options: logical selector length must match VariableNames",
                ));
            }
            Ok(mask
                .data
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| (*flag != 0).then_some(index))
                .collect())
        }
        _ => text_list(value)?
            .into_iter()
            .map(|name| {
                let mut found = names
                    .iter()
                    .enumerate()
                    .filter_map(|(index, candidate)| (candidate == &name).then_some(index));
                let index = found
                    .next()
                    .ok_or_else(|| invalid(format!("import options: unknown variable '{name}'")))?;
                if found.next().is_some() {
                    return Err(invalid(format!(
                        "import options: ambiguous variable name '{name}'"
                    )));
                }
                Ok(index)
            })
            .collect(),
    }
}

fn one_based_integer(value: &IntValue, len: usize) -> BuiltinResult<usize> {
    let index = value
        .try_to_usize()
        .ok_or_else(|| invalid("import options: invalid variable index"))?;
    if index == 0 || index > len {
        return Err(invalid(
            "import options: variable index exceeds NumVariables",
        ));
    }
    Ok(index - 1)
}

fn one_based_float(value: f64, len: usize) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > len as f64 {
        return Err(invalid("import options: invalid variable index"));
    }
    let index = value as usize;
    if index as f64 != value {
        return Err(invalid(
            "import options: variable index is not exactly representable",
        ));
    }
    Ok(index - 1)
}
