use super::assignment::assign_into_value;
use super::assignment::root;
use super::errors::{self, BUILTIN_NAME};
use super::selector::assign_with_selector;
use crate::builtins::introspection::dynamicprops;
use crate::builtins::structs::core::field_path::FieldStep;
use crate::execution::{RootedValue, RootedValueLease};
use crate::sequence::ResolveValueSequence;
use crate::{
    call_builtin_async, object_property_getter_name, object_property_setter_name, BuiltinResult,
};
use runmat_builtins::{SETFIELD_ERROR_OBJECT_PROPERTY, SETFIELD_OBJECT_FAMILY_EXTENSION};
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

pub(super) async fn assign_into_object(
    mut object: ObjectInstance,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _aggregate_root =
        RootedValueLease::for_traced(&object).map_err(|error| error.into_runtime_error())?;
    let _rhs_root =
        RootedValueLease::for_traced(&rhs).map_err(|error| error.into_runtime_error())?;
    ensure_object_family_enabled()?;
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = read_property(&object, &first.name).await?;
            assign_with_selector(current.value().clone(), selector, rest, rhs).await?
        } else {
            root(rhs)?
        };
        let setter_result =
            write_property(&mut object, &first.name, updated.value().clone()).await?;
        let output = root(Value::Object(object));
        drop(setter_result);
        return output;
    }
    let current = read_property(&object, &first.name).await?;
    let updated = if let Some(selector) = &first.index {
        assign_with_selector(current.value().clone(), selector, rest, rhs).await?
    } else {
        assign_into_value(current.value().clone(), rest, rhs).await?
    };
    let setter_result = write_property(&mut object, &first.name, updated.value().clone()).await?;
    let output = root(Value::Object(object));
    drop(setter_result);
    output
}

pub(super) fn ensure_object_family_enabled() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &SETFIELD_OBJECT_FAMILY_EXTENSION,
        BUILTIN_NAME,
    )
}

async fn read_property(obj: &ObjectInstance, name: &str) -> BuiltinResult<RootedValue> {
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

async fn write_property(
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
