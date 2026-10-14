use super::super::super::errors;
use crate::builtins::introspection::dynamicprops;
use crate::execution::RootedValue;
use crate::sequence::ResolveValueSequence;
use crate::{call_builtin_async, object_property_setter_name, BuiltinResult};
use runmat_builtins::SETFIELD_ERROR_OBJECT_PROPERTY;
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn write_property(
    obj: &mut ObjectInstance,
    name: &str,
    rhs: Value,
) -> BuiltinResult<Option<RootedValue>> {
    if obj.selected_class().is_some() {
        // Metadata writes can publish through a second handle before ordinary
        // property admission. Validate the receiver's exact selected product
        // before entering that side-effecting special case.
        let _ = crate::class_registry::checked_receiver_projection(obj)?;
    }
    if dynamicprops::metadata_assignment(obj, name, rhs.clone())? {
        return Ok(None);
    }
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if let Some((prop, owner)) = projection.receiver_property(&obj.class_name, &name.into()) {
            if prop.is_static {
                return Err(errors::static_access(format!(
                    "Property '{}' is static; use classref('{}').{}",
                    name, obj.class_name, name
                )));
            }
            if prop.is_constant
                || !projection.receiver_member_accessible(
                    &owner,
                    prop.set_access,
                    crate::dispatcher::class_access_context().as_ref(),
                )
            {
                return Err(errors::private_access(format!(
                    "Property '{name}' is not writable"
                )));
            }
            if prop.is_dependent {
                let setter = object_property_setter_name(name);
                if projection.receiver_declares_method(&obj.class_name, &setter.as_str().into()) {
                    let outputs =
                        crate::object::dispatch::call_object_property_setter_with_outputs(
                            Value::Object(obj.clone()),
                            name,
                            rhs,
                            1,
                        )
                        .await?;
                    let updated = outputs.try_map_owned(
                        |sequence| {
                            let mut values = sequence.resolve(
                                runmat_types::SequenceUse::RequireSingle,
                                crate::sequence::SequenceResolutionContext::default(),
                            )?;
                            values.pop().ok_or_else(|| {
                                errors::with_message(
                                    "dependent setter returned no value",
                                    &SETFIELD_ERROR_OBJECT_PROPERTY,
                                )
                            })
                        },
                        |error| error.into_runtime_error(),
                    )?;
                    if let Value::Object(updated_object) = updated.value() {
                        *obj = updated_object.clone();
                        return Ok(Some(updated));
                    }
                    return Err(errors::with_message(
                        format!(
                            "Dependent property setter for '{}' must return the updated object",
                            name
                        ),
                        &SETFIELD_ERROR_OBJECT_PROPERTY,
                    ));
                }
                return Err(errors::with_message(
                    format!("Dependent property '{name}' has no set method"),
                    &SETFIELD_ERROR_OBJECT_PROPERTY,
                ));
            }
            obj.properties.insert(name.to_string(), rhs);
            return Ok(None);
        }
        if dynamicprops::dynamic_property_assign(obj, name, rhs.clone())? {
            return Ok(None);
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
            return Err(errors::static_access(format!(
                "Property '{}' is static; use classref('{}').{}",
                name, obj.class_name, name
            )));
        }
        if prop.set_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
                "Property '{name}' is private"
            )));
        }
        if prop.is_dependent {
            let setter = object_property_setter_name(name);
            match call_builtin_async(&setter, &[Value::Object(obj.clone()), rhs.clone()]).await {
                Ok(updated) if matches!(updated.value(), Value::Object(_)) => {
                    if let Value::Object(updated_object) = updated.value() {
                        *obj = updated_object.clone();
                    }
                    return Ok(Some(updated));
                }
                Ok(_) => {
                    return Err(errors::with_message(
                        format!(
                            "Dependent property setter for '{}' must return the updated object",
                            name
                        ),
                        &SETFIELD_ERROR_OBJECT_PROPERTY,
                    ));
                }
                Err(err) if !errors::is_undefined_function(&err) => {
                    return Err(errors::remap(err, None));
                }
                Err(_) => {}
            }
            obj.properties.insert(format!("{name}_backing"), rhs);
            return Ok(None);
        }
    }
    if dynamicprops::dynamic_property_assign(obj, name, rhs.clone())? {
        return Ok(None);
    }
    obj.properties.insert(name.to_string(), rhs);
    Ok(None)
}
