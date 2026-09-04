use runmat_value::{CellArray, CharArray, Tensor, Value};

use crate::BuiltinResult;

use super::super::keys::GroupIndex;
use super::error;

pub(super) fn indices(index: &GroupIndex) -> BuiltinResult<Value> {
    Tensor::new(index.ids.clone(), vec![index.ids.len(), 1])
        .map(Value::Tensor)
        .map_err(error::internal)
}

pub(super) fn names(index: &GroupIndex) -> BuiltinResult<Value> {
    let values = index
        .keys
        .iter()
        .map(|key| {
            key.first()
                .map(|atom| Value::CharArray(CharArray::new_row(&atom.label())))
                .ok_or_else(|| error::internal("grp2idx: empty group key"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    CellArray::new(values, index.keys.len(), 1)
        .map(Value::Cell)
        .map_err(error::internal)
}
