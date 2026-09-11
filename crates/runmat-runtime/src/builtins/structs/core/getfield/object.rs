use super::errors;
use crate::builtins::introspection::dynamicprops;
use crate::{call_builtin_async, object_property_getter_name, BuiltinResult};
use runmat_builtins::{
    GETFIELD_ERROR_INVALID_HANDLE, GETFIELD_ERROR_MISSING_FIELD,
    GETFIELD_ERROR_NON_STRUCT_REFERENCE, GETFIELD_ERROR_OBJECT_PROPERTY,
    GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS, GETFIELD_OBJECT_FAMILY_EXTENSION,
};
use runmat_types::MemberAccess;
use runmat_value::{HandleRef, ObjectInstance, StructValue, Value};

#[async_recursion::async_recursion(?Send)]
pub(super) async fn get_field_value(
    value: Value,
    name: &str,
    enforce_public_builtin_extension: bool,
) -> BuiltinResult<Value> {
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
        Value::Object(obj) => get_object_field(&obj, name).await,
        Value::HandleObject(handle) => {
            get_handle_field(&handle, name, enforce_public_builtin_extension).await
        }
        Value::Listener(listener) => super::special::listener_field(&listener, name),
        Value::MException(ex) => super::special::exception_field(&ex, name),
        Value::StructArray(array) => {
            if array.is_empty() {
                return Err(errors::with_message(
                    "Struct contents reference from an empty struct array.",
                    &GETFIELD_ERROR_NON_STRUCT_REFERENCE,
                ));
            }
            array
                .get_linear(0)
                .and_then(|element| element.fields.get(name))
                .cloned()
                .ok_or_else(|| errors::from_descriptor(&GETFIELD_ERROR_MISSING_FIELD))
        }
        _ => Err(errors::from_descriptor(
            &GETFIELD_ERROR_NON_STRUCT_REFERENCE,
        )),
    }
}

fn get_struct_field(struct_value: &StructValue, name: &str) -> BuiltinResult<Value> {
    struct_value.fields.get(name).cloned().ok_or_else(|| {
        errors::with_message(
            format!("Reference to non-existent field '{}'.", name),
            &GETFIELD_ERROR_MISSING_FIELD,
        )
    })
}

async fn get_object_field(obj: &ObjectInstance, name: &str) -> BuiltinResult<Value> {
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(&obj.class_name, &name.into())
    {
        if prop.is_static {
            return Err(errors::with_message(
                format!(
                    "You cannot access the static property '{}' through an instance of class '{}'.",
                    name, obj.class_name
                ),
                &GETFIELD_ERROR_OBJECT_PROPERTY,
            ));
        }
        if prop.get_access == MemberAccess::Private {
            return Err(private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        if prop.is_dependent {
            let getter = object_property_getter_name(name);
            match call_builtin_async(&getter, &[Value::Object(obj.clone())]).await {
                Ok(value) => return Ok(value),
                Err(err) if !errors::is_undefined_function(&err) => {
                    return Err(errors::remap(err, None));
                }
                Err(_) => {}
            }
            if let Some(value) = obj.properties.get(&format!("{name}_backing")) {
                return Ok(value.clone());
            }
        }
    }
    if let Some(value) = dynamicprops::dynamic_property_read(obj, name)? {
        return Ok(value);
    }
    if let Some(value) = obj.properties.get(name) {
        return Ok(value.clone());
    }
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(&obj.class_name, &name.into())
    {
        if prop.get_access == MemberAccess::Private {
            return Err(private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        return Err(errors::with_message(
            format!(
                "No public property '{}' for class '{}'.",
                name, obj.class_name
            ),
            &GETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
    Err(errors::with_message(
        format!("Undefined property '{}' for class {}", name, obj.class_name),
        &GETFIELD_ERROR_OBJECT_PROPERTY,
    ))
}

fn private_access(message: impl Into<String>) -> crate::RuntimeError {
    errors::with_message(message, &GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS)
}

#[async_recursion::async_recursion(?Send)]
async fn get_handle_field(
    handle: &HandleRef,
    name: &str,
    enforce_public_builtin_extension: bool,
) -> BuiltinResult<Value> {
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
    get_field_value(target, name, enforce_public_builtin_extension).await
}
