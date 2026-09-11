use super::assignment::assign_into_value;
use super::errors::{self, BUILTIN_NAME};
use super::selector::assign_with_selector;
use crate::builtins::introspection::dynamicprops;
use crate::builtins::structs::core::field_path::FieldStep;
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
) -> BuiltinResult<Value> {
    ensure_object_family_enabled()?;
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = read_property(&object, &first.name).await?;
            assign_with_selector(current, selector, rest, rhs).await?
        } else {
            rhs
        };
        write_property(&mut object, &first.name, updated).await?;
        return Ok(Value::Object(object));
    }
    let current = read_property(&object, &first.name).await?;
    let updated = if let Some(selector) = &first.index {
        assign_with_selector(current, selector, rest, rhs).await?
    } else {
        assign_into_value(current, rest, rhs).await?
    };
    write_property(&mut object, &first.name, updated).await?;
    Ok(Value::Object(object))
}

pub(super) fn ensure_object_family_enabled() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &SETFIELD_OBJECT_FAMILY_EXTENSION,
        BUILTIN_NAME,
    )
}

async fn read_property(obj: &ObjectInstance, name: &str) -> BuiltinResult<Value> {
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
                return Ok(value.clone());
            }
        }
    }
    if let Some(value) = obj.properties.get(name) {
        return Ok(value.clone());
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

async fn write_property(obj: &mut ObjectInstance, name: &str, rhs: Value) -> BuiltinResult<()> {
    if dynamicprops::metadata_assignment(obj, name, rhs.clone())? {
        return Ok(());
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
                Ok(Value::Object(updated)) => {
                    *obj = updated;
                    return Ok(());
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
            return Ok(());
        }
    }
    if dynamicprops::dynamic_property_assign(obj, name, rhs.clone())? {
        return Ok(());
    }
    obj.properties.insert(name.to_string(), rhs);
    Ok(())
}
