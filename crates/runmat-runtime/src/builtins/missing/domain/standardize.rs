use super::numeric::char_rows;
use super::text_output::{selected_string_array, TextLiteral, TextSelection};
use super::*;

pub(super) struct IndicatorSet {
    numeric: Vec<NumericIndicator>,
    text: Vec<String>,
}

pub(super) enum NumericIndicator {
    Floating(f64),
    Integer(IntValue),
}

impl NumericIndicator {
    fn matches(&self, value: f64) -> bool {
        match self {
            Self::Floating(marker) => numeric_indicator_matches(value, *marker),
            Self::Integer(marker) => {
                crate::builtins::common::validation::integer_is_exact_f64(marker)
                    && value == marker.to_f64()
            }
        }
    }
}

pub(super) fn indicator_set(value: &Value) -> BuiltinResult<IndicatorSet> {
    let mut set = IndicatorSet {
        numeric: Vec::new(),
        text: Vec::new(),
    };
    collect_indicators(value, &mut set)?;
    Ok(set)
}

pub(super) fn collect_indicators(value: &Value, set: &mut IndicatorSet) -> BuiltinResult<()> {
    match value {
        Value::Num(n) => set.numeric.push(NumericIndicator::Floating(*n)),
        Value::Int(i) => set.numeric.push(NumericIndicator::Integer(i.clone())),
        Value::String(s) => set.text.push(s.as_str().to_owned()),
        Value::StringArray(array) => set
            .text
            .extend(array.data.iter().map(|text| text.as_str().to_owned())),
        Value::CharArray(array) => set.text.extend(char_rows(array)),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                for index in 0..storage.len() {
                    let marker = storage.value_at(index).ok_or_else(|| {
                        internal_error("standardizeMissing: integer indicator storage mismatch")
                    })?;
                    set.numeric.push(NumericIndicator::Integer(marker));
                }
            } else {
                set.numeric.extend(
                    tensor_utils::tensor_values_f64(tensor)
                        .into_iter()
                        .map(NumericIndicator::Floating),
                );
            }
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                collect_indicators(item, set)?;
            }
        }
        other => {
            return Err(unsupported_type(format!(
                "unsupported missing indicator {other:?}"
            )))
        }
    }
    Ok(())
}

pub(super) fn standardize_missing_value(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    indicators: &IndicatorSet,
) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => Ok(Value::Tensor(tensor)),
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            let shape = tensor.shape().to_vec();
            let mut values = tensor_utils::tensor_into_values_f64(tensor);
            for value in &mut values {
                if indicators
                    .numeric
                    .iter()
                    .any(|marker| marker.matches(*value))
                {
                    *value = f64::NAN;
                }
            }
            Tensor::new_with_dtype(values, shape, dtype)
                .map(Value::Tensor)
                .map_err(internal_error)
        }
        Value::String(mut s) => {
            if indicators
                .text
                .iter()
                .any(|marker| marker.as_str() == s.as_str())
            {
                s = MISSING_TEXT.into();
            }
            Ok(Value::String(s))
        }
        Value::StringArray(array) => {
            let output = selected_string_array(
                runtime,
                array,
                TextLiteral::Utf8(MISSING_TEXT),
                |array, index| {
                    if indicators
                        .text
                        .iter()
                        .any(|marker| marker.as_str() == array.data[index].as_str())
                    {
                        TextSelection::Literal
                    } else {
                        TextSelection::Input(index)
                    }
                },
                |runtime, cause| {
                    super::errors::allocation_error(runtime, "standardizeMissing", cause)
                },
                internal_error,
            )?;
            Ok(Value::StringArray(output))
        }
        Value::CharArray(array) => {
            let rows = char_rows(&array);
            let data: Vec<String> = rows
                .into_iter()
                .map(|text| {
                    if indicators.text.iter().any(|marker| marker == &text) {
                        MISSING_TEXT.to_string()
                    } else {
                        text
                    }
                })
                .collect();
            Ok(Value::StringArray(
                StringArray::new(data, vec![array.rows, 1]).map_err(internal_error)?,
            ))
        }
        Value::Object(mut object) if is_tabular_object(&object) => {
            let variables = table_variables(&object)?;
            let mut out = StructValue::new();
            for (name, field) in variables.fields {
                out.insert(name, standardize_missing_value(runtime, field, indicators)?);
            }
            object
                .properties
                .insert("__table_variables".to_string(), Value::Struct(out));
            Ok(Value::Object(object))
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            let data = match cell.data.try_into_owned() {
                Ok(values) => values.into_iter().collect::<Vec<_>>(),
                Err(shared) => shared.iter().cloned().collect::<Vec<_>>(),
            };
            for item in data {
                out.push(standardize_missing_value(runtime, item, indicators)?);
            }
            Ok(Value::Cell(
                CellArray::new(out, cell.rows, cell.cols).map_err(internal_error)?,
            ))
        }
        other => Ok(other),
    }
}

pub(super) fn numeric_indicator_matches(value: f64, marker: f64) -> bool {
    if marker.is_nan() {
        value.is_nan()
    } else {
        value == marker
    }
}

pub(super) fn numeric_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(internal_error),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(internal_error),
        Value::LogicalArray(array) => Tensor::new(
            array
                .data
                .iter()
                .map(|flag| f64::from(*flag != 0))
                .collect(),
            array.shape,
        )
        .map_err(internal_error),
        other => Err(unsupported_type(format!(
            "{context}: expected numeric input, got {other:?}"
        ))),
    }
}
