use super::errors;
use super::object;
use super::selector::assign_with_selector;
use crate::builtins::structs::core::field_path::{FieldStep, IndexComponent, IndexSelector};
use crate::BuiltinResult;
use runmat_builtins::{
    SETFIELD_ERROR_FIELD_EXPECTED, SETFIELD_ERROR_INDEX_SHAPE, SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
};
use runmat_value::{StructValue, Value};

pub(super) async fn assign_value(
    base: Value,
    leading_index: Option<IndexSelector>,
    steps: Vec<FieldStep>,
    rhs: Value,
) -> BuiltinResult<Value> {
    if steps.is_empty() {
        return Err(errors::with_message(
            SETFIELD_ERROR_FIELD_EXPECTED.message,
            &SETFIELD_ERROR_FIELD_EXPECTED,
        ));
    }
    if let Some(selector) = leading_index {
        assign_with_leading_index(base, &selector, &steps, rhs).await
    } else {
        assign_without_leading_index(base, &steps, rhs).await
    }
}

async fn assign_with_leading_index(
    base: Value,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    match base {
        Value::StructArray(array) => {
            super::struct_array::assign_into_struct_array(array, selector, steps, rhs).await
        }
        other => Err(errors::with_message(
            format!("setfield: leading indices require a struct array, got {other:?}"),
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
    }
}

async fn assign_without_leading_index(
    base: Value,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    match base {
        Value::Struct(struct_value) => assign_into_struct(struct_value, steps, rhs).await,
        Value::Object(object) => object::assign_into_object(object, steps, rhs).await,
        Value::StructArray(array) if array.is_empty() => Err(errors::with_message(
            "setfield: struct array is empty; supply indices in a cell array",
            &SETFIELD_ERROR_INDEX_SHAPE,
        )),
        Value::StructArray(array) => {
            let selector = IndexSelector {
                components: vec![IndexComponent::Scalar(1)],
            };
            super::struct_array::assign_into_struct_array(array, &selector, steps, rhs).await
        }
        Value::HandleObject(handle) => super::handle::assign_into_handle(handle, steps, rhs).await,
        Value::Listener(_) => Err(errors::with_message(
            "setfield: listeners do not support direct field assignment",
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
        other => Err(errors::with_message(
            format!(
                "setfield unsupported on this value for field '{}': {other:?}",
                steps
                    .first()
                    .map(|step| step.name.as_str())
                    .unwrap_or_default()
            ),
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn assign_into_value(
    value: Value,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    if steps.is_empty() {
        return Ok(rhs);
    }
    match value {
        Value::Struct(struct_value) => assign_into_struct(struct_value, steps, rhs).await,
        Value::StructArray(array) => {
            let selector = IndexSelector {
                components: vec![IndexComponent::Scalar(1)],
            };
            super::struct_array::assign_into_struct_array(array, &selector, steps, rhs).await
        }
        Value::Object(object) => object::assign_into_object(object, steps, rhs).await,
        Value::HandleObject(handle) => super::handle::assign_into_handle(handle, steps, rhs).await,
        Value::Listener(_) => Err(errors::with_message(
            "setfield: listeners do not support nested field assignment",
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
        other => Err(errors::with_message(
            format!("Struct contents assignment to a {other:?} object is not supported."),
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn assign_into_struct(
    mut struct_value: StructValue,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = struct_value
                .fields
                .get(&first.name)
                .cloned()
                .ok_or_else(|| {
                    errors::with_message(
                        format!("Reference to non-existent field '{}'.", first.name),
                        &runmat_builtins::SETFIELD_ERROR_MISSING_FIELD,
                    )
                })?;
            assign_with_selector(current, selector, rest, rhs).await?
        } else {
            rhs
        };
        struct_value.fields.insert(first.name.clone(), updated);
        return Ok(Value::Struct(struct_value));
    }
    if let Some(selector) = &first.index {
        let current = struct_value
            .fields
            .get(&first.name)
            .cloned()
            .ok_or_else(|| {
                errors::with_message(
                    format!("Reference to non-existent field '{}'.", first.name),
                    &runmat_builtins::SETFIELD_ERROR_MISSING_FIELD,
                )
            })?;
        let updated = assign_with_selector(current, selector, rest, rhs).await?;
        struct_value.fields.insert(first.name.clone(), updated);
        return Ok(Value::Struct(struct_value));
    }
    let current = struct_value
        .fields
        .get(&first.name)
        .cloned()
        .unwrap_or_else(|| Value::Struct(StructValue::new()));
    let updated = assign_into_value(current, rest, rhs).await?;
    struct_value.fields.insert(first.name.clone(), updated);
    Ok(Value::Struct(struct_value))
}
