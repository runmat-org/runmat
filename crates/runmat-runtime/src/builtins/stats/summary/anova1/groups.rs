use super::*;
use crate::builtins::common::grouping_keys::KeyAtom;
use std::collections::BTreeMap;

pub(super) struct Groups {
    pub names: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

pub(super) async fn collect(
    y: &Tensor,
    group: Option<&Value>,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Groups> {
    if y.shape().iter().skip(2).any(|&extent| extent != 1) || y.is_empty() {
        return Err(invalid(NAME, "y must be a nonempty vector or matrix"));
    }
    let rows = y.shape().first().copied().unwrap_or(1);
    let columns = y.shape().get(1).copied().unwrap_or(1);
    let vector = rows == 1 || columns == 1;
    if vector && group.is_none() {
        return Err(invalid(
            NAME,
            "vector observations require a grouping vector",
        ));
    }
    let expected = if vector { y.len() } else { columns };
    let mut labels = Vec::with_capacity(expected);
    match group {
        None => {
            for index in 0..expected {
                labels.push(Some(KeyAtom::Text((index + 1).to_string())));
            }
        }
        Some(value) => {
            let count = label_count(value)?;
            if count != expected {
                return Err(invalid(NAME,"group must contain one label per observation for vectors, or per column for matrices"));
            }
            for index in 0..expected {
                labels.push(label_at(value, index)?);
                if index % 256 == 255 {
                    cooperation.completed(256).await?;
                }
            }
            cooperation.completed(expected % 256).await?;
        }
    }
    let mut slots = BTreeMap::<KeyAtom, usize>::new();
    let mut names = Vec::new();
    let mut values: Vec<Vec<f64>> = Vec::new();
    for (index, label) in labels.into_iter().enumerate() {
        let Some(label) = label else { continue };
        let slot = if let Some(&slot) = slots.get(&label) {
            slot
        } else {
            let slot = names.len();
            names.push(label.label());
            values.push(Vec::new());
            slots.insert(label, slot);
            slot
        };
        let extent = if vector { 1 } else { rows };
        let base = if vector { index } else { index * rows };
        for offset in 0..extent {
            let value = crate::builtins::common::tensor::tensor_value_f64(y, base + offset);
            if !value.is_nan() {
                values[slot].push(value);
            }
            if offset % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(extent % 1024).await?;
    }
    let mut data = Groups {
        names: Vec::new(),
        values: Vec::new(),
    };
    for (name, values) in names.into_iter().zip(values) {
        if !values.is_empty() {
            data.names.push(name);
            data.values.push(values);
        }
    }
    if data.values.len() < 2 {
        return Err(invalid(
            NAME,
            "at least two nonempty observed groups are required",
        ));
    }
    Ok(data)
}

fn label_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::String(_) | Value::Num(_) | Value::Bool(_) => Ok(1),
        Value::CharArray(value) => Ok(value.rows),
        Value::StringArray(value) => Ok(value.data().len()),
        Value::Cell(value) => Ok(value.data().len()),
        Value::LogicalArray(value) => Ok(value.data.len()),
        Value::Tensor(value)
            if value.shape().iter().filter(|&&n| n != 1).count() <= 1
                && matches!(
                    value.numeric_dtype(),
                    runmat_value::NumericDType::F64 | runmat_value::NumericDType::F32
                ) =>
        {
            Ok(value.len())
        }
        Value::Object(value) if value.is_class(runmat_types::standard::CATEGORICAL) => {
            Ok(crate::builtins::table::categorical_label_view(value)?.len())
        }
        _ => Err(invalid(
            NAME,
            "group must be a numeric/logical vector, text labels, cellstr, or categorical vector",
        )),
    }
}
fn text(value: &str) -> Option<KeyAtom> {
    if value.is_empty() {
        None
    } else {
        Some(KeyAtom::Text(value.to_owned()))
    }
}
fn label_at(value: &Value, index: usize) -> BuiltinResult<Option<KeyAtom>> {
    Ok(match value {
        Value::String(value) => value.as_str().and_then(text),
        Value::StringArray(value) => value.data()[index].as_str().and_then(text),
        Value::CharArray(value) => {
            let codes = (0..value.cols)
                .map(|column| value.data()[index * value.cols + column])
                .collect::<Vec<_>>();
            let value = runmat_value::decode_utf16_code_units(&codes).map_err(|source| {
                crate::build_runtime_error("anova1: invalid group text")
                    .with_source(source)
                    .build()
            })?;
            text(value.trim_end())
        }
        Value::Cell(value) => {
            let value =
                crate::builtins::common::validation::value_text_scalar_string(&value.data()[index])
                    .ok_or_else(|| {
                        invalid(NAME, "cell grouping input must contain text scalar labels")
                    })?;
            text(&value)
        }
        Value::Bool(value) => Some(KeyAtom::Logical(*value)),
        Value::LogicalArray(value) => Some(KeyAtom::Logical(value.data[index] != 0)),
        Value::Num(value) => KeyAtom::from_numeric(runmat_value::NumericScalar::F64(*value)),
        Value::Tensor(value) => value
            .numeric_value_at(index)
            .and_then(KeyAtom::from_numeric),
        Value::Object(value) => crate::builtins::table::categorical_label_view(value)?
            .observation_label_at(index)?
            .and_then(text),
        _ => return Err(invalid(NAME, "unsupported group class")),
    })
}
