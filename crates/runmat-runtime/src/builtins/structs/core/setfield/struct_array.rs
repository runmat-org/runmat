use super::assignment::assign_into_value;
use super::errors;
use crate::builtins::structs::core::field_path::{self, FieldStep, IndexSelector};
use crate::BuiltinResult;
use runmat_builtins::{SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS, SETFIELD_ERROR_INDEX_SHAPE};
use runmat_value::{StructArray, Tensor, Value};

pub(super) async fn assign_into_struct_array(
    array: StructArray,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    let plan = field_path::build_plan(&Value::StructArray(array.clone()), selector, false)
        .map_err(|error| errors::remap_index(error, Some("setfield: ")))?;
    let [position] = plan.indices.as_slice() else {
        return Err(errors::with_message(
            "setfield: leading indices must select one structure-array element",
            &SETFIELD_ERROR_INDEX_SHAPE,
        ));
    };
    let position = *position as usize;
    let current = array
        .get_linear(position)
        .map(|element| element.to_owned())
        .ok_or_else(|| {
            errors::with_message(
                SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS.message,
                &SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
            )
        })?;
    let updated = assign_into_value(Value::Struct(current), steps, rhs).await?;
    let Value::Struct(updated) = updated else {
        return Err(errors::internal(
            "structure-array element update did not produce a structure",
        ));
    };
    let mut array = array;
    let new_fields = updated
        .field_names()
        .filter(|name| !array.field_names().any(|existing| existing == *name))
        .cloned()
        .collect::<Vec<_>>();
    for field in new_fields {
        array
            .insert_field(
                field,
                vec![Value::Tensor(Tensor::zeros(vec![0, 0])); array.len()],
            )
            .map_err(errors::internal)?;
    }
    array = array
        .replace_linear_scalar(&[position], updated)
        .map_err(errors::internal)?;
    Ok(Value::StructArray(array))
}
