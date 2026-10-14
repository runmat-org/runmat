use super::errors;
use crate::builtins::introspection::dynamicprops;
use crate::execution::{RootedValue, RootedValueLease};
use crate::sequence::ResolveValueSequence;
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
        Value::Object(obj) => get_object_field(runtime, &obj, name).await,
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

async fn get_object_field(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &ObjectInstance,
    name: &str,
) -> BuiltinResult<RootedValue> {
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        let property = projection.receiver_property(&obj.class_name, &name.into());
        if let Some((prop, owner)) = &property {
            if prop.is_static {
                return Err(errors::with_message(
                    format!("You cannot access the static property '{}' through an instance of class '{}'.", name, obj.class_name),
                    &GETFIELD_ERROR_OBJECT_PROPERTY,
                ));
            }
            if !projection.receiver_member_accessible(
                owner,
                prop.get_access,
                crate::dispatcher::class_access_context().as_ref(),
            ) {
                return Err(private_access(format!(
                    "You cannot get the '{}' property of '{}' class.",
                    name, obj.class_name
                )));
            }
            if prop.is_dependent {
                let getter = object_property_getter_name(name);
                if projection.receiver_declares_method(&obj.class_name, &getter.as_str().into()) {
                    let outputs = match runtime {
                        Some(runtime) => crate::object::dispatch::call_object_property_getter_with_outputs_in_context(
                            runtime, Value::Object(obj.clone()), name, 1,
                        ).await,
                        None => crate::object::dispatch::call_object_property_getter_with_outputs(
                            Value::Object(obj.clone()), name, 1,
                        ).await,
                    }?;
                    return outputs.try_map_owned(
                        |sequence| {
                            let mut values = sequence.resolve(
                                runmat_types::SequenceUse::RequireSingle,
                                crate::sequence::SequenceResolutionContext::default(),
                            )?;
                            values.pop().ok_or_else(|| {
                                errors::with_message(
                                    "dependent getter returned no value",
                                    &GETFIELD_ERROR_OBJECT_PROPERTY,
                                )
                            })
                        },
                        |error| error.into_runtime_error(),
                    );
                }
                if let Some(value) = obj.properties.get(&format!("{name}_backing")) {
                    return RootedValue::new(value.clone())
                        .map_err(|error| error.into_runtime_error());
                }
            }
        }
        if let Some(value) = dynamicprops::dynamic_property_read(obj, name)? {
            return RootedValue::new(value).map_err(|error| error.into_runtime_error());
        }
        if let Some(value) = obj.properties.get(name) {
            return RootedValue::new(value.clone()).map_err(|error| error.into_runtime_error());
        }
        return Err(errors::with_message(
            format!("Undefined property '{}' for class {}", name, obj.class_name),
            &GETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
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
            let receiver = [Value::Object(obj.clone())];
            let result = match runtime {
                Some(runtime) => {
                    crate::call_builtin_async_in_context(runtime, &getter, &receiver).await
                }
                None => call_builtin_async(&getter, &receiver).await,
            };
            match result {
                Ok(value) => return Ok(value),
                Err(err) if !errors::is_undefined_function(&err) => {
                    return Err(errors::remap(err, None));
                }
                Err(_) => {}
            }
            if let Some(value) = obj.properties.get(&format!("{name}_backing")) {
                return RootedValue::new(value.clone()).map_err(|error| error.into_runtime_error());
            }
        }
    }
    if let Some(value) = dynamicprops::dynamic_property_read(obj, name)? {
        return RootedValue::new(value).map_err(|error| error.into_runtime_error());
    }
    if let Some(value) = obj.properties.get(name) {
        return RootedValue::new(value.clone()).map_err(|error| error.into_runtime_error());
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
    get_field_value(runtime, target, name, enforce_public_builtin_extension).await
}
