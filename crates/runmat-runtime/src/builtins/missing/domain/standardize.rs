use super::numeric::char_rows;
use super::text_output::{selected_string_array, TextLiteral, TextSelection};
use super::*;

pub(super) struct IndicatorSet {
    numeric: Vec<NumericIndicator>,
    text: Vec<runmat_value::StringElement>,
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
        Value::String(s) => set.text.push(s.clone()),
        Value::StringArray(array) => set.text.extend(array.data().iter().cloned()),
        Value::CharArray(array) => set
            .text
            .extend(char_rows(array)?.into_iter().map(Into::into)),
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
            for item in cell.data() {
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
            if indicators.text.iter().any(|marker| marker == &s) {
                s = runmat_value::StringElement::Missing;
            }
            Ok(Value::String(s.into()))
        }
        Value::StringArray(array) => {
            let output = selected_string_array(
                runtime,
                array,
                TextLiteral::Missing,
                |array, index| {
                    if indicators
                        .text
                        .iter()
                        .any(|marker| marker == &array.data()[index])
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
            let rows = char_rows(&array)?;
            let data: Vec<runmat_value::StringElement> = rows
                .into_iter()
                .map(|text| {
                    if indicators
                        .text
                        .iter()
                        .any(|marker| marker.as_str() == Some(text.as_str()))
                    {
                        runmat_value::StringElement::Missing
                    } else {
                        text.into()
                    }
                })
                .collect();
            Ok(Value::StringArray(
                StringArray::from_elements_unbounded(data, vec![array.rows, 1])
                    .map_err(internal_error)?,
            ))
        }
        Value::Object(mut object) if is_tabular_object(&object) => {
            let variables = table_variables(&object)?;
            let mut out = runmat_value::UnboundedStructValueBuilder::new();
            for (name, field) in variables.into_fields_unbounded() {
                out.insert(name, standardize_missing_value(runtime, field, indicators)?);
            }
            crate::execution::object::replacement::replace_property(
                runtime,
                &mut object,
                "__table_variables",
                Value::Struct(out.finish()),
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        runtime,
                        "RunMat:standardizeMissing:Allocation",
                        message,
                        source,
                    )
                },
                internal_error,
            )?;
            Ok(Value::Object(object))
        }
        Value::Cell(cell) => {
            let (mut cell, _roots) =
                crate::execution::structure::replacement::cell(&cell, runtime)?;
            let values = cell.try_data_mut_unique().ok_or_else(|| {
                internal_error("standardizeMissing: prepared Cell owner is unexpectedly shared")
            })?;
            for item in values {
                let source = std::mem::replace(item, Value::Num(0.0));
                *item = standardize_missing_value(runtime, source, indicators)?;
            }
            Ok(Value::Cell(cell))
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
