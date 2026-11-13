use runmat_value::{LogicalArray, NumericStorage, Tensor, Value};

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
            .map_err(super::VariableError::from)?;
        return GroupColumn::vector(name, Value::Tensor(value)).map(|column| vec![column]);
    }
    let rows = value.rows();
    let columns = value.cols();
    (0..columns)
        .map(|column| {
            let mut storage = NumericStorage::zeros(value.numeric_dtype(), rows);
            for row in 0..rows {
                let element = value.numeric_value_at(row + column * rows).ok_or_else(|| {
                    super::VariableError::from("numeric array shape does not match its storage")
                })?;
                storage
                    .set_value(row, element)
                    .map_err(super::VariableError::from)?;
            }
            let value = Tensor::from_numeric_storage(storage, vec![rows, 1])
                .map_err(super::VariableError::from)?;
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
            .map_err(super::VariableError::from)?;
        return GroupColumn::vector(name, Value::LogicalArray(value)).map(|column| vec![column]);
    }
    (0..columns)
        .map(|column| {
            let data = (0..rows)
                .map(|row| {
                    value.data.get(row + column * rows).copied().ok_or_else(|| {
                        super::VariableError::from("logical array shape does not match its storage")
                    })
                })
                .collect::<VariableResult<Vec<_>>>()?;
            let value =
                LogicalArray::new(data, vec![rows, 1]).map_err(super::VariableError::from)?;
            GroupColumn::vector(format!("{name}{}", column + 1), Value::LogicalArray(value))
        })
        .collect()
}
