use super::*;

pub(in crate::builtins::table) fn split_value_columns(value: Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Tensor(
                    Tensor::new_with_dtype(data, vec![rows, 1], tensor.dtype)
                        .map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::ComplexTensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.cols);
            for col in 0..tensor.cols {
                let mut data = Vec::with_capacity(tensor.rows);
                for row in 0..tensor.rows {
                    data.push(tensor.data[row + col * tensor.rows]);
                }
                out.push(Value::ComplexTensor(
                    ComplexTensor::new(data, vec![tensor.rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::StringArray(array) => {
            let rows = array.rows();
            let cols = array.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(array.data[row + col * rows].clone());
                }
                out.push(Value::StringArray(
                    StringArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::LogicalArray(array) => {
            let rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(*array.data.get(row + col * rows).ok_or_else(|| {
                        invalid_variable("array2table: logical array shape mismatch")
                    })?);
                }
                out.push(Value::LogicalArray(
                    LogicalArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.cols);
            for col in 0..cell.cols {
                let mut data = Vec::with_capacity(cell.rows);
                for row in 0..cell.rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Cell(
                    CellArray::new(data, cell.rows, 1).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        other => Ok(vec![other]),
    }
}
