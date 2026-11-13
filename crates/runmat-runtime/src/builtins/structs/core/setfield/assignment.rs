use super::errors;
use super::object;
use super::selector::assign_with_selector;
use crate::builtins::structs::core::field_path::{FieldStep, IndexComponent, IndexSelector};
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{
    SETFIELD_ERROR_FIELD_EXPECTED, SETFIELD_ERROR_INDEX_SHAPE, SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
};
use runmat_value::Value;

mod structure;

pub(super) async fn assign_value(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    leading_index: Option<IndexSelector>,
    steps: Vec<FieldStep>,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _base_root = super::super::construction::input(runtime, &base)?;
    let _rhs_root = super::super::construction::input(runtime, &rhs)?;
    if steps.is_empty() {
        return Err(errors::with_message(
            SETFIELD_ERROR_FIELD_EXPECTED.message,
            &SETFIELD_ERROR_FIELD_EXPECTED,
        ));
    }
    if let Some(selector) = leading_index {
        assign_with_leading_index(runtime, base, &selector, &steps, rhs).await
    } else {
        assign_without_leading_index(runtime, base, &steps, rhs).await
    }
}

async fn assign_with_leading_index(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    match base {
        Value::StructArray(array) => {
            super::struct_array::assign_into_struct_array(runtime, array, selector, steps, rhs)
                .await
        }
        other => Err(errors::with_message(
            format!("setfield: leading indices require a struct array, got {other:?}"),
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        )),
    }
}

async fn assign_without_leading_index(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    match base {
        Value::Struct(struct_value) => {
            structure::assign_into_struct(runtime, struct_value, steps, rhs).await
        }
        Value::Object(object) => object::assign_into_object(runtime, object, steps, rhs).await,
        Value::StructArray(array) if array.is_empty() => Err(errors::with_message(
            "setfield: struct array is empty; supply indices in a cell array",
            &SETFIELD_ERROR_INDEX_SHAPE,
        )),
        Value::StructArray(array) => {
            let selector = IndexSelector {
                components: vec![IndexComponent::Scalar(1)],
            };
            super::struct_array::assign_into_struct_array(runtime, array, &selector, steps, rhs)
                .await
        }
        Value::HandleObject(handle) => {
            super::handle::assign_into_handle(runtime, handle, steps, rhs).await
        }
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
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    if steps.is_empty() {
        return super::super::construction::root(runtime, rhs);
    }
    match value {
        Value::Struct(struct_value) => {
            structure::assign_into_struct(runtime, struct_value, steps, rhs).await
        }
        Value::StructArray(array) => {
            let selector = IndexSelector {
                components: vec![IndexComponent::Scalar(1)],
            };
            super::struct_array::assign_into_struct_array(runtime, array, &selector, steps, rhs)
                .await
        }
        Value::Object(object) => object::assign_into_object(runtime, object, steps, rhs).await,
        Value::HandleObject(handle) => {
            super::handle::assign_into_handle(runtime, handle, steps, rhs).await
        }
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
