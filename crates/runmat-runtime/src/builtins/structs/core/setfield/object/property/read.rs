use super::super::super::errors;
use crate::execution::RootedValue;
use crate::sequence::ResolveValueSequence;
use crate::{call_builtin_async, object_property_getter_name, BuiltinResult};
use runmat_builtins::SETFIELD_ERROR_OBJECT_PROPERTY;
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

fn root(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<RootedValue> {
    crate::builtins::structs::core::construction::root(runtime, value)
}

async fn invoke_getter(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &ObjectInstance,
    name: &str,
) -> BuiltinResult<RootedValue> {
    let getter = object_property_getter_name(name);
    match runtime {
        Some(runtime) => {
            let outputs =
                crate::object::dispatch::call_object_property_getter_with_outputs_in_context(
                    runtime,
                    Value::Object(obj.clone()),
                    name,
                    1,
                )
                .await?;
            outputs.try_map_owned(
                |sequence| {
                    let values = sequence.resolve(
                        runmat_types::SequenceUse::RequireSingle,
                        crate::sequence::SequenceResolutionContext::default(),
                    )?;
                    crate::sequence::into_first_output(values)?.ok_or_else(|| {
                        errors::with_message(
                            "dependent getter returned no value",
                            &SETFIELD_ERROR_OBJECT_PROPERTY,
                        )
                    })
                },
                |error| error.into_runtime_error(),
            )
        }
        None => call_builtin_async(&getter, &[Value::Object(obj.clone())]).await,
    }
}

pub(in super::super) async fn read_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &ObjectInstance,
    name: &str,
) -> BuiltinResult<RootedValue> {
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if let Some((prop, owner)) = projection.receiver_property(obj.class_name(), &name.into()) {
            if prop.is_static {
                return Err(errors::with_message(
                    format!("You cannot access the static property '{}' through an instance of class '{}'.", name, obj.class_name()),
                    &runmat_builtins::SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
                ));
            }
            if !projection.receiver_member_accessible(
                &owner,
                prop.get_access,
                crate::dispatcher::class_access_context().as_deref(),
            ) {
                return Err(errors::private_access(format!(
                    "You cannot get the '{}' property of '{}' class.",
                    name,
                    obj.class_name()
                )));
            }
            if prop.is_dependent {
                let getter = object_property_getter_name(name);
                if projection.receiver_declares_method(obj.class_name(), &getter.as_str().into()) {
                    return invoke_getter(runtime, obj, name).await;
                }
                if let Some(value) = obj.property(&format!("{name}_backing")) {
                    return root(runtime, value.clone());
                }
            }
        }
        if let Some(value) = obj.property(name) {
            return root(runtime, value.clone());
        }
        return Err(errors::with_message(
            format!(
                "Undefined property '{}' for class {}",
                name,
                obj.class_name()
            ),
            &SETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(obj.class_name(), &name.into())
    {
        if prop.is_static {
            return Err(errors::with_message(
                format!(
                    "You cannot access the static property '{}' through an instance of class '{}'.",
                    name,
                    obj.class_name()
                ),
                &runmat_builtins::SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
            ));
        }
        if prop.get_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name,
                obj.class_name()
            )));
        }
        if prop.is_dependent {
            match invoke_getter(runtime, obj, name).await {
                Ok(value) => return Ok(value),
                Err(err) if !errors::is_undefined_function(&err) => {
                    return Err(errors::remap(err, None));
                }
                Err(_) => {}
            }
            if let Some(value) = obj.property(&format!("{name}_backing")) {
                return root(runtime, value.clone());
            }
        }
    }
    if let Some(value) = obj.property(name) {
        return root(runtime, value.clone());
    }
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(obj.class_name(), &name.into())
    {
        if prop.get_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name,
                obj.class_name()
            )));
        }
        return Err(errors::with_message(
            format!(
                "No public property '{}' for class '{}'.",
                name,
                obj.class_name()
            ),
            &SETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
    Err(errors::with_message(
        format!(
            "Undefined property '{}' for class {}",
            name,
            obj.class_name()
        ),
        &SETFIELD_ERROR_OBJECT_PROPERTY,
    ))
}
