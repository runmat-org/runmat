use runmat_value::{SparseTensor, Tensor, Value};

use crate::BuiltinResult;

use super::{error, shape};

pub(super) fn numeric(
    data: Vec<f64>,
    dimensions: Vec<usize>,
    sparse: bool,
) -> BuiltinResult<Value> {
    if !sparse {
        return Tensor::new(data, dimensions)
            .map(Value::Tensor)
            .map_err(error::invalid);
    }
    if dimensions.len() > 2 {
        return Err(error::invalid(
            "accumarray: sparse output is only supported for 2-D results",
        ));
    }
    let (rows, columns) = shape::rows_and_columns(&dimensions);
    let mut column_starts = Vec::with_capacity(columns + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    column_starts.push(0);
    for column in 0..columns {
        for row in 0..rows {
            let value = data[row + column * rows];
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        column_starts.push(values.len());
    }
    SparseTensor::new(rows, columns, column_starts, row_indices, values)
        .map(Value::SparseTensor)
        .map_err(error::invalid)
}
