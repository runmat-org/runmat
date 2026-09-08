use runmat_value::{CellArray, StructValue, Value};

pub(super) fn parse(value: &Value) -> crate::BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Struct(structure) => Ok(Some(field_order(structure))),
        Value::Cell(array) if starts_with_structure(array) => parse_array(array).map(Some),
        _ => Ok(None),
    }
}

fn starts_with_structure(array: &CellArray) -> bool {
    matches!(array.iter_column_major().next(), Some(Value::Struct(_)))
}

fn parse_array(array: &CellArray) -> crate::BuiltinResult<Vec<String>> {
    let mut structures = array.iter_column_major().enumerate();
    let Some((_, Value::Struct(first))) = structures.next() else {
        return Ok(Vec::new());
    };
    let order = field_order(first);
    for (index, value) in structures {
        let Value::Struct(structure) = value else {
            return Err(super::super::error::invalid_reference(format!(
                "element {} is not a structure",
                index + 1
            )));
        };
        super::validation::exact_field_set(order.as_slice(), field_order(structure).as_slice())
            .map_err(|_| {
                super::super::error::invalid_reference(format!(
                    "element {} has a different field schema",
                    index + 1
                ))
            })?;
    }
    Ok(order)
}

fn field_order(structure: &StructValue) -> Vec<String> {
    structure.field_names().cloned().collect()
}
