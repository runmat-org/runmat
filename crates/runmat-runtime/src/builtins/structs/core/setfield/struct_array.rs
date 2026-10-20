use super::assignment::assign_into_value;
use super::errors;
use crate::builtins::structs::core::construction;
use crate::builtins::structs::core::field_path::{self, FieldStep, IndexSelector};
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS, SETFIELD_ERROR_INDEX_SHAPE};
use runmat_value::{StructArray, Value};

pub(super) async fn assign_into_struct_array(
    runtime: Option<&crate::context::RuntimeContext>,
    array: StructArray,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _aggregate_root = construction::input(runtime, &Value::StructArray(array.clone()))?;
    let _rhs_root = construction::input(runtime, &rhs)?;
    let plan = field_path::build_plan(&Value::StructArray(array.clone()), selector, false)
        .map_err(|error| errors::remap_index(error, Some("setfield: ")))?;
    let [position] = plan.indices.as_slice() else {
        return Err(errors::with_message(
            "setfield: leading indices must select one structure-array element",
            &SETFIELD_ERROR_INDEX_SHAPE,
        ));
    };
    let position = *position as usize;
    let current = array.get_linear(position).ok_or_else(|| {
        errors::with_message(
            SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS.message,
            &SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
        )
    })?;
    let current = construction::scalar(runtime, current.fields.iter())?;
    let updated = assign_into_value(runtime, current.value().clone(), steps, rhs).await?;
    let Value::Struct(structure) = updated.value() else {
        return Err(errors::internal(
            "structure-array element update did not produce a structure",
        ));
    };
    // A path edits one top-level field. Preserve every other physical column;
    // only the changed column needs new, admitted Value slots.
    let field = &steps.first().expect("assignment path was validated").name;
    let replacement = structure
        .fields
        .get(field)
        .ok_or_else(|| errors::internal("structure-array update lost the assigned field"))?;
    let old_column = array.column_owner(field);
    let default = if old_column.is_none() {
        Some(construction::empty_array(runtime)?)
    } else {
        None
    };
    let column = construction::column(runtime, array.len(), |index| {
        if index == position {
            replacement.clone()
        } else if let Some(column) = old_column {
            column.as_slice()[index].clone()
        } else {
            default
                .as_ref()
                .expect("new field has an empty default")
                .value()
                .clone()
        }
    })?;
    column.with_owned_output_storage(|column| {
        let columns = array
            .column_owners()
            .map(|(name, original)| (name, if name == field { &column } else { original }))
            .chain(old_column.is_none().then_some((field.as_str(), &column)));
        construction::array(runtime, columns, array.shape_owner())
    })
}
