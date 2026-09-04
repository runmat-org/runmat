use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;

use super::{GroupColumn, VariableResult};

pub(super) fn tensor_columns(
    name: &str,
    value: Tensor,
    split: bool,
) -> VariableResult<Vec<GroupColumn>> {
    if !split || value.cols() <= 1 || value.rows() == 1 {
        let rows = tensor_utils::tensor_element_len(&value);
        let value = value
            .reshape(vec![rows, 1])
            .map_err(|error| error.to_string())?;
        return GroupColumn::vector(name, Value::Tensor(value)).map(|column| vec![column]);
    }
    let rows = value.rows();
    let columns = value.cols();
    let storage = value
        .into_numeric_storage()
        .map_err(|error| error.to_string())?;
    (0..columns)
        .map(|column| {
            let indices = (0..rows).map(|row| row + column * rows).collect::<Vec<_>>();
            let storage = storage
                .gather(&indices)
                .map_err(|error| error.to_string())?;
            let value = Tensor::from_numeric_storage(storage, vec![rows, 1])
                .map_err(|error| error.to_string())?;
            GroupColumn::vector(format!("{name}{}", column + 1), Value::Tensor(value))
        })
        .collect()
}

pub(super) fn logical_columns(
    name: &str,
    value: LogicalArray,
    split: bool,
) -> VariableResult<Vec<GroupColumn>> {
    let rows = value.shape.first().copied().unwrap_or(value.data.len());
    let columns = value.shape.get(1).copied().unwrap_or(1);
    if !split || columns <= 1 || rows == 1 {
        let rows = value.data.len();
        let value = LogicalArray::from_host_buffer(value.data, vec![rows, 1])
            .map_err(|error| error.to_string())?;
        return GroupColumn::vector(name, Value::LogicalArray(value)).map(|column| vec![column]);
    }
    (0..columns)
        .map(|column| {
            let data =
                (0..rows)
                    .map(|row| {
                        value.data.get(row + column * rows).copied().ok_or_else(|| {
                            "logical array shape does not match its storage".to_string()
                        })
                    })
                    .collect::<VariableResult<Vec<_>>>()?;
            let value =
                LogicalArray::new(data, vec![rows, 1]).map_err(|error| error.to_string())?;
            GroupColumn::vector(format!("{name}{}", column + 1), Value::LogicalArray(value))
        })
        .collect()
}
