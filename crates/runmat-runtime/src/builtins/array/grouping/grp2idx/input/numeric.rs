use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::{ensure_vector, numeric_atom, GroupingInput};
use crate::builtins::array::grouping::grp2idx::error;
use crate::builtins::array::grouping::keys::KeyOrder;

pub(super) fn tensor(value: Tensor) -> BuiltinResult<GroupingInput> {
    ensure_vector(&value.shape, "numeric")?;
    let len = tensor_utils::tensor_element_len(&value);
    let rows = rows(&value)?;
    let value = value.reshape(vec![len, 1]).map_err(error::invalid)?;
    Ok(GroupingInput::new(
        Value::Tensor(value),
        rows,
        KeyOrder::Sorted,
    ))
}

pub(super) fn logical(value: LogicalArray) -> BuiltinResult<GroupingInput> {
    ensure_vector(&value.shape, "logical")?;
    let len = value.data.len();
    let rows = value
        .data
        .iter()
        .map(|value| Some(vec![super::KeyAtom::Logical(*value != 0)]))
        .collect();
    let value = LogicalArray::from_host_buffer(value.data, vec![len, 1]).map_err(error::invalid)?;
    Ok(GroupingInput::new(
        Value::LogicalArray(value),
        rows,
        KeyOrder::Sorted,
    ))
}

pub(super) fn rows(value: &Tensor) -> BuiltinResult<Vec<Option<Vec<super::KeyAtom>>>> {
    (0..tensor_utils::tensor_element_len(value))
        .map(|index| {
            value
                .numeric_value_at(index)
                .map(numeric_atom)
                .ok_or_else(|| error::invalid("grp2idx: numeric storage is malformed"))
        })
        .collect()
}
