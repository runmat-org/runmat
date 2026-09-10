use crate::sequence::ValueSequence;
use crate::RuntimeError;
use runmat_value::{StructArray, Value};

pub fn gather_member(array: StructArray, field: &str) -> Result<ValueSequence, RuntimeError> {
    let mut array = array;
    let values = array
        .remove_field(field)
        .ok_or_else(|| RuntimeError::from(format!("Undefined field '{field}'")))?;
    Ok(ValueSequence::comma_separated(values))
}

pub fn assign_member_values<OnWrite>(
    array: StructArray,
    field: String,
    values: Vec<Value>,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    if values.len() != array.len() {
        return Err(format!(
            "structure-array field assignment requires exactly one value per destination element (expected {}, received {})",
            array.len(),
            values.len()
        )
        .into());
    }
    let mut array = array;
    if array.field_names().any(|name| name == &field) {
        array
            .replace_field_values(&field, values, &mut on_write)
            .map_err(RuntimeError::from)?;
    } else {
        array
            .insert_field(field, values)
            .map_err(RuntimeError::from)?;
    }
    Ok(Value::StructArray(array))
}
