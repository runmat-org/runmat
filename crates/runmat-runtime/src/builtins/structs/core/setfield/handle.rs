use super::assignment::assign_into_value;
use super::errors;
use super::object::ensure_object_family_enabled;
use crate::builtins::structs::core::field_path::FieldStep;
use crate::BuiltinResult;
use runmat_builtins::{
    SETFIELD_ERROR_INVALID_HANDLE, SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
    SETFIELD_ERROR_OBJECT_PROPERTY,
};
use runmat_value::{HandleRef, Value};

pub(super) async fn assign_into_handle(
    handle: HandleRef,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<Value> {
    ensure_object_family_enabled()?;
    if steps.is_empty() {
        return Err(errors::with_message(
            "setfield: expected at least one field name when assigning into a handle",
            &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        ));
    }
    if !crate::is_handle_valid(&handle) {
        return Err(invalid_handle(&handle, None));
    }
    let current = runmat_gc::gc_clone_value(&handle.target)
        .map_err(|error| invalid_handle(&handle, Some(error.to_string())))?;
    let updated = assign_into_value(current.clone(), steps, rhs).await?;
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        let target_valid = match target {
            Value::Object(obj) => !matches!(
                obj.properties.get(crate::HANDLE_VALID_FLAG_PROPERTY),
                Some(Value::Bool(false))
            ),
            _ => false,
        };
        if !target_valid {
            return Err(invalid_handle(&handle, None));
        }
        if *target != current {
            return Err(errors::with_message(
                "setfield: handle target changed during asynchronous assignment",
                &SETFIELD_ERROR_OBJECT_PROPERTY,
            ));
        }
        runmat_gc::gc_record_handle_write(&handle.target, &updated);
        *target = updated;
        Ok(())
    })
    .map_err(|error| invalid_handle(&handle, Some(error.to_string())))??;
    Ok(Value::HandleObject(handle))
}

fn invalid_handle(handle: &HandleRef, detail: Option<String>) -> crate::RuntimeError {
    let message = detail.map_or_else(
        || format!("Invalid or deleted handle object '{}'.", handle.class_name),
        |detail| format!("setfield: invalid handle target: {detail}"),
    );
    errors::with_message(message, &SETFIELD_ERROR_INVALID_HANDLE)
}
