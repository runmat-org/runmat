use runmat_value::{CellArray, CharArray, Value};

use crate::builtins::table::select_rows;
use crate::BuiltinResult;

use super::GroupingInput;
use crate::builtins::array::grouping::grp2idx::error;
use crate::builtins::array::grouping::keys::GroupIndex;

pub(super) fn build(input: &GroupingInput, index: &GroupIndex) -> BuiltinResult<Value> {
    if input.categorical {
        let Value::Object(object) = &input.value else {
            return Err(error::internal(
                "grp2idx: categorical input state is malformed",
            ));
        };
        return crate::builtins::table::categorical_declared_levels(object);
    }
    if matches!(input.value, Value::StringArray(_) | Value::String(_)) {
        return cellstr(index);
    }
    let rows = index
        .first_rows
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| error::internal("grp2idx: observed level has no source row"))?;
    select_rows(&input.value, &rows).map_err(|source| {
        error::internal(format!(
            "grp2idx: could not construct group levels: {source}"
        ))
    })
}

fn cellstr(index: &GroupIndex) -> BuiltinResult<Value> {
    let values = index
        .keys
        .iter()
        .map(|key| Value::CharArray(CharArray::new_row(&key[0].label())))
        .collect();
    CellArray::new(values, index.keys.len(), 1)
        .map(Value::Cell)
        .map_err(error::internal)
}
