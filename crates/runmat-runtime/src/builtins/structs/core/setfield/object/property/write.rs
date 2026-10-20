use super::super::super::errors;
use crate::builtins::introspection::dynamicprops;
use crate::context::host_allocation::{host_allocation_runtime_error, HostAllocationError};
use crate::execution::RootedValue;
use crate::{object_property_setter_name, BuiltinResult};
use runmat_builtins::SETFIELD_ERROR_OBJECT_PROPERTY;
use runmat_types::MemberAccess;
use runmat_value::{ObjectInstance, Value};

mod setter;

fn allocation_error(
    runtime: Option<&crate::context::RuntimeContext>,
    message: &'static str,
    source: HostAllocationError,
) -> crate::RuntimeError {
    errors::remap(
        host_allocation_runtime_error(runtime, "RunMat:setfield:ObjectAllocation", message, source),
        None,
    )
}

fn replace_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &mut ObjectInstance,
    name: &str,
    value: Value,
) -> BuiltinResult<Option<Value>> {
    crate::execution::object::replacement::replace_property(
        runtime,
        obj,
        name,
        value,
        |message, source| allocation_error(runtime, message, source),
        |message| errors::with_message(message, &SETFIELD_ERROR_OBJECT_PROPERTY),
    )
}

pub(in super::super) async fn write_property(
    runtime: Option<&crate::context::RuntimeContext>,
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
    if dynamicprops::metadata_assignment(runtime, obj, name, rhs.clone())? {
        return Ok(None);
    }
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if let Some((prop, owner)) = projection.receiver_property(obj.class_name(), &name.into()) {
            if prop.is_static {
                return Err(errors::static_access(format!(
                    "Property '{}' is static; use classref('{}').{}",
                    name,
                    obj.class_name(),
                    name
                )));
            }
            if prop.is_constant
                || !projection.receiver_member_accessible(
                    &owner,
                    prop.set_access,
                    crate::dispatcher::class_access_context().as_deref(),
                )
            {
                return Err(errors::private_access(format!(
                    "Property '{name}' is not writable"
                )));
            }
            if prop.is_dependent {
                let setter = object_property_setter_name(name);
                if projection.receiver_declares_method(obj.class_name(), &setter.as_str().into()) {
                    let updated = setter::invoke(runtime, obj, name, rhs).await?;
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
            let old = replace_property(runtime, obj, name, rhs)?;
            drop(old);
            return Ok(None);
        }
        if let Some(permitted) = dynamicprops::dynamic_property_assignment_access(obj, name, None) {
            if !permitted {
                return Err(errors::private_access(format!(
                    "Property '{name}' is not writable"
                )));
            }
            let old = replace_property(runtime, obj, name, rhs)?;
            drop(old);
            return Ok(None);
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
            return Err(errors::static_access(format!(
                "Property '{}' is static; use classref('{}').{}",
                name,
                obj.class_name(),
                name
            )));
        }
        if prop.set_access == MemberAccess::Private {
            return Err(errors::private_access(format!(
                "Property '{name}' is private"
            )));
        }
        if prop.is_dependent {
            match setter::invoke(runtime, obj, name, rhs.clone()).await {
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
            let backing = format!("{name}_backing");
            let old = replace_property(runtime, obj, &backing, rhs)?;
            drop(old);
            return Ok(None);
        }
    }
    if let Some(permitted) = dynamicprops::dynamic_property_assignment_access(obj, name, None) {
        if !permitted {
            return Err(errors::private_access(format!(
                "Property '{name}' is private"
            )));
        }
    }
    let old = replace_property(runtime, obj, name, rhs)?;
    drop(old);
    Ok(None)
}
