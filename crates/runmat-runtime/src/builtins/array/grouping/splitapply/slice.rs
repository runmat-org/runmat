use runmat_value::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::table::{select_rows, value_row_count};
use crate::BuiltinResult;

use super::error;
use super::groups::{Groups, SplitAxis};

pub(super) fn validate_observation_count(value: &Value, groups: &Groups) -> BuiltinResult<()> {
    let actual = match groups.axis {
        SplitAxis::Rows => value_row_count(value)?,
        SplitAxis::Columns => column_count(value)?,
    };
    if actual == groups.observation_count {
        Ok(())
    } else {
        Err(error::invalid(format!(
            "splitapply: data has {actual} observations along the split dimension; G has {}",
            groups.observation_count
        )))
    }
}

pub(super) fn select(value: &Value, groups: &Groups, indices: &[usize]) -> BuiltinResult<Value> {
    match groups.axis {
        SplitAxis::Rows => select_rows(value, indices).map_err(|error| {
            super::error::invalid(format!("splitapply: cannot select data rows: {error}"))
        }),
        SplitAxis::Columns => select_columns(value, indices),
    }
}

fn column_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(value) => Ok(value.shape.get(1).copied().unwrap_or(1)),
        Value::ComplexTensor(value) => Ok(value.shape.get(1).copied().unwrap_or(1)),
        Value::LogicalArray(value) => Ok(value.shape.get(1).copied().unwrap_or(1)),
        Value::StringArray(value) => Ok(value.cols()),
        Value::Cell(value) => Ok(value.cols),
        Value::CharArray(value) => Ok(value.cols),
        _ => Err(error::invalid(
            "splitapply: row-oriented G requires array data with a second dimension",
        )),
    }
}

fn select_columns(value: &Value, columns: &[usize]) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(value) => numeric_columns(value, columns),
        Value::ComplexTensor(value) => complex_columns(value, columns),
        Value::LogicalArray(value) => logical_columns(value, columns),
        Value::StringArray(value) => string_columns(value, columns),
        Value::Cell(value) => cell_columns(value, columns),
        Value::CharArray(value) => char_columns(value, columns),
        _ => Err(error::invalid(
            "splitapply: this data representation cannot be split by columns",
        )),
    }
}

fn column_indices(shape: &[usize], selected: &[usize]) -> BuiltinResult<(Vec<usize>, Vec<usize>)> {
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    let planes = shape.get(2..).unwrap_or_default().iter().product::<usize>();
    let mut indices = Vec::with_capacity(rows * selected.len() * planes);
    for plane in 0..planes {
        for column in selected {
            if *column >= columns {
                return Err(error::invalid(
                    "splitapply: data column index is out of bounds",
                ));
            }
            let start = (plane * columns + *column) * rows;
            indices.extend(start..start + rows);
        }
    }
    let mut output_shape = if shape.is_empty() {
        vec![1, selected.len()]
    } else {
        shape.to_vec()
    };
    if output_shape.len() == 1 {
        output_shape.push(selected.len());
    } else {
        output_shape[1] = selected.len();
    }
    Ok((indices, output_shape))
}

fn numeric_columns(value: &Tensor, columns: &[usize]) -> BuiltinResult<Value> {
    let (indices, shape) = column_indices(&value.shape, columns)?;
    let storage = value
        .clone()
        .into_numeric_storage()
        .map_err(error::invalid)?
        .gather(&indices)
        .map_err(error::invalid)?;
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(error::invalid)
}

fn complex_columns(value: &ComplexTensor, columns: &[usize]) -> BuiltinResult<Value> {
    let (indices, shape) = column_indices(&value.shape, columns)?;
    let storage = value
        .complex_storage()
        .gather(&indices)
        .map_err(error::invalid)?;
    ComplexTensor::from_complex_storage(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(error::invalid)
}

fn logical_columns(value: &LogicalArray, columns: &[usize]) -> BuiltinResult<Value> {
    let (indices, shape) = column_indices(&value.shape, columns)?;
    let data = indices
        .into_iter()
        .map(|index| {
            value.data.get(index).copied().ok_or_else(|| {
                error::invalid("splitapply: logical data storage is inconsistent with its shape")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(error::invalid)
}

fn string_columns(value: &StringArray, columns: &[usize]) -> BuiltinResult<Value> {
    let shape = vec![value.rows(), value.cols()];
    let (indices, output_shape) = column_indices(&shape, columns)?;
    let data = indices
        .into_iter()
        .map(|index| {
            value.data.get(index).cloned().ok_or_else(|| {
                error::invalid("splitapply: string data storage is inconsistent with its shape")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    StringArray::new(data, output_shape)
        .map(Value::StringArray)
        .map_err(error::invalid)
}

fn cell_columns(value: &CellArray, columns: &[usize]) -> BuiltinResult<Value> {
    let (indices, _) = column_indices(&[value.rows, value.cols], columns)?;
    let data = indices
        .into_iter()
        .map(|index| {
            value.data.get(index).cloned().ok_or_else(|| {
                error::invalid("splitapply: cell data storage is inconsistent with its shape")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    CellArray::new(data, value.rows, columns.len())
        .map(Value::Cell)
        .map_err(error::invalid)
}

fn char_columns(value: &CharArray, columns: &[usize]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(value.rows * columns.len());
    for row in 0..value.rows {
        for column in columns {
            if *column >= value.cols {
                return Err(error::invalid(
                    "splitapply: character data column is out of bounds",
                ));
            }
            data.push(value.data[row * value.cols + *column]);
        }
    }
    CharArray::new(data, value.rows, columns.len())
        .map(Value::CharArray)
        .map_err(error::invalid)
}
