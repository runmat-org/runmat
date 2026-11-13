mod property;

use super::errors;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{
    GETFIELD_ERROR_INVALID_HANDLE, GETFIELD_ERROR_MISSING_FIELD,
    GETFIELD_ERROR_NON_STRUCT_REFERENCE, GETFIELD_OBJECT_FAMILY_EXTENSION,
};
use runmat_value::{HandleRef, StructValue, Value};

#[async_recursion::async_recursion(?Send)]
pub(super) async fn get_field_value(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    name: &str,
    enforce_public_builtin_extension: bool,
) -> BuiltinResult<RootedValue> {
    let _input_root = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|error| error.into_runtime_error())?;
    if enforce_public_builtin_extension
        && matches!(
            &value,
            Value::Object(_) | Value::HandleObject(_) | Value::Listener(_) | Value::MException(_)
        )
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GETFIELD_OBJECT_FAMILY_EXTENSION,
            errors::BUILTIN_NAME,
        )?;
    }
    match value {
        Value::Struct(st) => get_struct_field(&st, name),
        Value::Object(obj) => property::get_object_field(runtime, &obj, name, None).await,
        Value::HandleObject(handle) => {
            get_handle_field(runtime, &handle, name, enforce_public_builtin_extension).await
        }
        Value::Listener(listener) => {
            RootedValue::new(super::special::listener_field(&listener, name)?)
                .map_err(|error| error.into_runtime_error())
        }
        Value::MException(ex) => RootedValue::new(super::special::exception_field(&ex, name)?)
            .map_err(|error| error.into_runtime_error()),
        Value::StructArray(array) => {
            if array.is_empty() {
                return Err(errors::with_message(
                    "Struct contents reference from an empty struct array.",
                    &GETFIELD_ERROR_NON_STRUCT_REFERENCE,
                ));
            }
            RootedValue::new(
                array
                    .get_linear(0)
                    .and_then(|element| element.fields.get(name))
                    .cloned()
                    .ok_or_else(|| errors::from_descriptor(&GETFIELD_ERROR_MISSING_FIELD))?,
            )
            .map_err(|error| error.into_runtime_error())
        }
        _ => Err(errors::from_descriptor(
            &GETFIELD_ERROR_NON_STRUCT_REFERENCE,
        )),
    }
}

fn get_struct_field(struct_value: &StructValue, name: &str) -> BuiltinResult<RootedValue> {
    RootedValue::new(struct_value.fields.get(name).cloned().ok_or_else(|| {
        errors::with_message(
            format!("Reference to non-existent field '{}'.", name),
            &GETFIELD_ERROR_MISSING_FIELD,
        )
    })?)
    .map_err(|error| error.into_runtime_error())
}

#[async_recursion::async_recursion(?Send)]
async fn get_handle_field(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &HandleRef,
    name: &str,
    enforce_public_builtin_extension: bool,
) -> BuiltinResult<RootedValue> {
    if !crate::is_handle_valid(handle) {
        return Err(errors::with_message(
            format!("Invalid or deleted handle object '{}'.", handle.class_name),
            &GETFIELD_ERROR_INVALID_HANDLE,
        ));
    }
    let target = runmat_gc::gc_clone_value(&handle.target).map_err(|error| {
        errors::with_message(
            format!("getfield: invalid handle target: {error}"),
            &GETFIELD_ERROR_INVALID_HANDLE,
        )
    })?;
    match target {
        // Metadata and stored fields belong to the target, but a dependent
        // getter must receive the original issued handle, not its value-object
        // backing. The caller's root lease retains that identity while awaited.
        Value::Object(obj) => property::get_object_field(runtime, &obj, name, Some(handle)).await,
        target => get_field_value(runtime, target, name, enforce_public_builtin_extension).await,
    }
}
