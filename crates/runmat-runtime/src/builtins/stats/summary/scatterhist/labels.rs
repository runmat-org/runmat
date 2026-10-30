//! Scatterhist grouping: typed identity until presentation.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_value::NumericScalar;

pub(super) fn preflight_group_labels(value: &Value, expected_len: usize) -> BuiltinResult<()> {
    let Value::Cell(cell) = value else {
        return Ok(());
    };
    preflight_cell_labels(cell, expected_len)
}

fn preflight_cell_labels(cell: &CellArray, expected_len: usize) -> BuiltinResult<()> {
    if cell.data().len() != expected_len {
        return Err(invalid(
            "scatterhist: Group must have one label per observation",
        ));
    }
    for label in cell.data() {
        let scalar = match label {
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => true,
            Value::CharArray(chars) => chars.rows == 1,
            Value::StringArray(array) => array.data().len() == 1,
            Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 1,
            Value::LogicalArray(array) => array.len() == 1,
            Value::GpuTensor(handle) => {
                handle.physical().storage() == GpuTensorStorage::Real
                    && handle
                        .shape()
                        .iter()
                        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                        == Some(1)
            }
            _ => false,
        };
        if !scalar {
            return Err(invalid("scatterhist: cell Group labels must be scalar"));
        }
    }
    Ok(())
}

pub(super) fn group_labels(
    value: &Value,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<KeyAtom>>> {
    match value {
        Value::Num(value) if expected_len == 1 => {
            Ok(vec![scatterhist_numeric_key(NumericScalar::F64(*value))])
        }
        Value::Int(value) if expected_len == 1 => {
            Ok(vec![Some(KeyAtom::Integer(value.clone()))])
        }
        Value::Bool(value) if expected_len == 1 => {
            Ok(vec![Some(KeyAtom::Logical(*value))])
        }
        Value::String(value) if expected_len == 1 => {
            Ok(vec![value.as_str().map(|text| KeyAtom::Text(text.to_owned()))])
        }
        Value::LogicalArray(array) if array.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|value| Some(KeyAtom::Logical(*value != 0)))
            .collect()),
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == expected_len => {
            Ok((0..expected_len)
                .map(|index| tensor.numeric_value_at(index).and_then(scatterhist_numeric_key))
                .collect())
        }
        Value::StringArray(array) if array.data().len() == expected_len => Ok(array
            .data()
            .iter()
            .map(|value| value.as_str().map(|text| KeyAtom::Text(text.to_owned())))
            .collect()),
        Value::CharArray(chars) if chars.rows == expected_len => (0..chars.rows)
            .map(|row| -> BuiltinResult<Option<KeyAtom>> {
                let start = row * chars.cols;
                let end = start + chars.cols;
                let text = crate::builtins::stats::decode_utf16_with_error(
                    &chars.data()[start..end],
                    || invalid("scatterhist: Group label is not valid UTF-16 text"),
                )?;
                Ok(Some(KeyAtom::Text(text.trim().to_string())))
            })
            .collect(),
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let labels = crate::builtins::table::categorical_observation_labels(object)
                .map_err(|source| invalid_with_source("scatterhist: invalid categorical Group value", source))?;
            if labels.len() != expected_len {
                return Err(invalid("scatterhist: Group must have one label per observation"));
            }
            Ok(labels.into_iter().map(|label| label.map(KeyAtom::Text)).collect())
        }
        _ => Err(invalid(
            "scatterhist: Group must have one numeric, text, cell, or categorical label per observation",
        )),
    }
}

pub(super) fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<KeyAtom>>> {
    preflight_cell_labels(cell, expected_len)?;
    cell.data().iter().map(scalar_group_label).collect()
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<KeyAtom>> {
    match value {
        Value::Num(value) => Ok(scatterhist_numeric_key(NumericScalar::F64(*value))),
        Value::Int(value) => Ok(Some(KeyAtom::Integer(value.clone()))),
        Value::Bool(value) => Ok(Some(KeyAtom::Logical(*value))),
        Value::String(value) => Ok(value.as_str().map(|text| KeyAtom::Text(text.to_owned()))),
        Value::CharArray(chars) => Ok(Some(KeyAtom::Text(
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid("scatterhist: Group label is not valid UTF-16 text")
            })?,
        ))),
        Value::StringArray(array) => Ok(array.data()[0]
            .as_str()
            .map(|text| KeyAtom::Text(text.to_owned()))),
        Value::Tensor(tensor) => Ok(tensor.numeric_value_at(0).and_then(scatterhist_numeric_key)),
        Value::LogicalArray(array) => Ok(Some(KeyAtom::Logical(array.data[0] != 0))),
        _ => Err(invalid("scatterhist: cell Group labels must be scalar")),
    }
}

fn scatterhist_numeric_key(value: NumericScalar) -> Option<KeyAtom> {
    match value {
        NumericScalar::F64(value) if !value.is_finite() => None,
        NumericScalar::F32(value) if !value.is_finite() => None,
        value => KeyAtom::from_numeric(value),
    }
}

pub(super) fn integer_values(tensor: &Tensor) -> Option<Vec<IntValue>> {
    tensor
        .integer_storage()
        .map(|storage| storage.exact_values())
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}
