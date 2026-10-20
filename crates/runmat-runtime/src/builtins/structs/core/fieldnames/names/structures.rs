use runmat_value::{StructArray, StructValue};

pub(super) fn scalar(structure: &StructValue) -> Vec<String> {
    structure.field_names().map(str::to_owned).collect()
}

pub(super) fn array(array: &StructArray) -> Vec<String> {
    array.field_names().map(str::to_owned).collect()
}
