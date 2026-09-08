use runmat_value::{CellArray, StructValue, Value};
use std::collections::BTreeSet;

pub(super) fn scalar(structure: &StructValue) -> Vec<String> {
    structure.field_names().cloned().collect()
}

pub(super) fn array(array: &CellArray) -> crate::BuiltinResult<Vec<String>> {
    let mut names = BTreeSet::new();
    for value in &array.data {
        let Value::Struct(structure) = value else {
            return Err(super::super::error::invalid_struct_array());
        };
        names.extend(structure.field_names().cloned());
    }
    Ok(names.into_iter().collect())
}
