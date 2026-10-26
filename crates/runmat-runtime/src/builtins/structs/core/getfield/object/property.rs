use super::super::errors;
use crate::builtins::introspection::dynamicprops;
use crate::execution::RootedValue;
use crate::sequence::ResolveValueSequence;
use crate::{call_builtin_async, object_property_getter_name, BuiltinResult};
use runmat_builtins::{GETFIELD_ERROR_OBJECT_PROPERTY, GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS};
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn get_object_field(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &ObjectInstance,
    name: &str,
    receiver: Option<&runmat_value::HandleRef>,
) -> BuiltinResult<RootedValue> {
    if receiver.is_none() {
        if let Some(prepared) = crate::builtins::structs::core::field_path::native::prepare(
            runtime,
            obj,
            crate::object::protocol::ObjectProtocol::Subsref,
            name,
        )? {
            return prepared.read(obj, name).await;
        }
    }
    let receiver = || match receiver {
        Some(handle) => Value::HandleObject(handle.clone()),
        None => Value::Object(obj.clone()),
    };
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        let property = projection.receiver_property(obj.class_name(), &name.into());
        if let Some((prop, owner)) = &property {
            if prop.is_static {
                return Err(errors::with_message(
                    format!("You cannot access the static property '{}' through an instance of class '{}'.", name, obj.class_name()),
                    &GETFIELD_ERROR_OBJECT_PROPERTY,
                ));
            }
            if !projection.receiver_member_accessible(
                owner,
                prop.get_access,
                crate::dispatcher::class_access_context().as_deref(),
            ) {
                return Err(private_access(format!(
                    "You cannot get the '{}' property of '{}' class.",
                    name,
                    obj.class_name()
                )));
            }
            if prop.is_dependent {
                let getter = object_property_getter_name(name);
                if projection.receiver_declares_method(obj.class_name(), &getter.as_str().into()) {
                    let outputs = match runtime {
                        Some(runtime) => crate::object::dispatch::call_object_property_getter_with_outputs_in_context(
                            runtime, receiver(), name, 1,
                        ).await,
                        None => crate::object::dispatch::call_object_property_getter_with_outputs(
                            receiver(), name, 1,
                        ).await,
                    }?;
                    return outputs.try_map_owned(
                        |sequence| {
                            let values = sequence.resolve(
                                runmat_types::SequenceUse::RequireSingle,
                                crate::sequence::SequenceResolutionContext::default(),
                            )?;
                            crate::sequence::into_first_output(values)?.ok_or_else(|| {
                                errors::with_message(
                                    "dependent getter returned no value",
                                    &GETFIELD_ERROR_OBJECT_PROPERTY,
                                )
                            })
                        },
                        |error| error.into_runtime_error(),
                    );
                }
                if let Some(value) = obj.properties().get(&format!("{name}_backing")) {
                    return crate::builtins::structs::core::construction::root(
                        runtime,
                        value.clone(),
                    );
                }
            }
        }
        if let Some(value) = dynamicprops::dynamic_property_read(obj, name)? {
            return crate::builtins::structs::core::construction::root(runtime, value);
        }
        if let Some(value) = obj.properties().get(name) {
            return crate::builtins::structs::core::construction::root(runtime, value.clone());
        }
        return Err(errors::with_message(
            format!(
                "Undefined property '{}' for class {}",
                name,
                obj.class_name()
            ),
            &GETFIELD_ERROR_OBJECT_PROPERTY,
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
                &GETFIELD_ERROR_OBJECT_PROPERTY,
            ));
        }
        if prop.get_access == MemberAccess::Private {
            return Err(private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name,
                obj.class_name()
            )));
        }
        if prop.is_dependent {
            let getter = object_property_getter_name(name);
            let receiver = [receiver()];
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
            if let Some(value) = obj.properties().get(&format!("{name}_backing")) {
                return crate::builtins::structs::core::construction::root(runtime, value.clone());
            }
        }
    }
    if let Some(value) = dynamicprops::dynamic_property_read(obj, name)? {
        return crate::builtins::structs::core::construction::root(runtime, value);
    }
    if let Some(value) = obj.properties().get(name) {
        return crate::builtins::structs::core::construction::root(runtime, value.clone());
    }
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(obj.class_name(), &name.into())
    {
        if prop.get_access == MemberAccess::Private {
            return Err(private_access(format!(
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
            &GETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
    Err(errors::with_message(
        format!(
            "Undefined property '{}' for class {}",
            name,
            obj.class_name()
        ),
        &GETFIELD_ERROR_OBJECT_PROPERTY,
    ))
}

fn private_access(message: impl Into<String>) -> crate::RuntimeError {
    errors::with_message(message, &GETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS)
}
