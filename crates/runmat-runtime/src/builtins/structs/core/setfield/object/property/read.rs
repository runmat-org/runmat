use super::super::super::assignment::root;
use super::super::super::errors;
use crate::execution::RootedValue;
use crate::sequence::ResolveValueSequence;
use crate::{call_builtin_async, object_property_getter_name, BuiltinResult};
use runmat_builtins::SETFIELD_ERROR_OBJECT_PROPERTY;
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn read_property(obj: &ObjectInstance, name: &str) -> BuiltinResult<RootedValue> {
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if let Some((prop, owner)) = projection.receiver_property(&obj.class_name, &name.into()) {
            if prop.is_static {
                return Err(errors::with_message(
                    format!("You cannot access the static property '{}' through an instance of class '{}'.", name, obj.class_name),
                    &runmat_builtins::SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
                ));
            }
            if !projection.receiver_member_accessible(
                &owner,
                prop.get_access,
                crate::dispatcher::class_access_context().as_ref(),
            ) {
                return Err(errors::private_access(format!(
                    "You cannot get the '{}' property of '{}' class.",
                    name, obj.class_name
                )));
            }
            if prop.is_dependent {
                let getter = object_property_getter_name(name);
                if projection.receiver_declares_method(&obj.class_name, &getter.as_str().into()) {
                    let outputs =
                        crate::object::dispatch::call_object_property_getter_with_outputs(
                            Value::Object(obj.clone()),
                            name,
                            1,
                        )
                        .await?;
                    return outputs.try_map_owned(
                        |sequence| {
                            let mut values = sequence.resolve(
                                runmat_types::SequenceUse::RequireSingle,
                                crate::sequence::SequenceResolutionContext::default(),
                            )?;
                            values.pop().ok_or_else(|| {
                                errors::with_message(
                                    "dependent getter returned no value",
                                    &SETFIELD_ERROR_OBJECT_PROPERTY,
                                )
                            })
                        },
                        |error| error.into_runtime_error(),
                    );
                }
                if let Some(value) = obj.properties.get(&format!("{name}_backing")) {
                    return root(value.clone());
                }
            }
        }
        if let Some(value) = obj.properties.get(name) {
            return root(value.clone());
        }
        return Err(errors::with_message(
            format!("Undefined property '{}' for class {}", name, obj.class_name),
            &SETFIELD_ERROR_OBJECT_PROPERTY,
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
                &runmat_builtins::SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
            ));
        }
        if prop.get_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
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
                return root(value.clone());
            }
        }
    }
    if let Some(value) = obj.properties.get(name) {
        return root(value.clone());
    }
    if let Some((prop, _owner)) =
        crate::class_registry::lookup_property(&obj.class_name, &name.into())
    {
        if prop.get_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        return Err(errors::with_message(
            format!(
                "No public property '{}' for class '{}'.",
                name, obj.class_name
            ),
            &SETFIELD_ERROR_OBJECT_PROPERTY,
        ));
    }
    Err(errors::with_message(
        format!("Undefined property '{}' for class {}", name, obj.class_name),
        &SETFIELD_ERROR_OBJECT_PROPERTY,
    ))
}
