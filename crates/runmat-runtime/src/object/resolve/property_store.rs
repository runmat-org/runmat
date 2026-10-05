//! Default property stores after the caller has resolved subsasgn exactly once.
//! Metadata/access admission is shared; value objects return replacements while
//! handle objects publish to their authentic target or invoke an in-place setter.
use super::*;
use runmat_value::{HandleRef, ObjectInstance};

enum Store {
    Stored,
    Dependent,
}

fn admit(obj: &ObjectInstance, field: &str, caller: Option<&str>) -> Result<Store, RuntimeError> {
    if let Some((property, owner)) =
        crate::class_registry::lookup_property(&obj.class_name, &field.into())
    {
        if property.is_static {
            return Err(mex(
                "RunMat:PropertyStaticAccess",
                &format!(
                    "Property '{field}' is static; use classref('{}').{field}",
                    obj.class_name
                ),
            ));
        }
        if property.is_constant {
            return Err(mex(
                IDENT_PROPERTY_READ_ONLY,
                &format!("Property '{field}' is constant"),
            ));
        }
        if !access_permitted(&owner, &property.set_access, caller) {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{field}' is private"),
            ));
        }
        if property.is_dependent {
            if crate::class_registry::lookup_method(
                &obj.class_name,
                &crate::object_property_setter_name(field).as_str().into(),
            )
            .is_some()
            {
                return Ok(Store::Dependent);
            }
            return Err(mex(
                IDENT_PROPERTY_READ_ONLY,
                &format!("Dependent property '{field}' has no set method"),
            ));
        }
        return Ok(Store::Stored);
    }
    if let Some(permitted) = dynamicprops::dynamic_property_assignment_access(
        obj,
        field,
        caller_class_for_function(caller).as_ref(),
    ) {
        if !permitted {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{field}' is private"),
            ));
        }
        return Ok(Store::Stored);
    }
    if crate::class_registry::get_class(&obj.class_name).is_some() {
        Err(format!("Undefined property '{field}' for class {}", obj.class_name).into())
    } else {
        Err(format!("Unknown class {}", obj.class_name).into())
    }
}

pub(super) async fn store_value(
    context: Option<&crate::context::RuntimeContext>,
    mut obj: ObjectInstance,
    field: String,
    rhs: Value,
    caller: Option<&str>,
    on_write: &mut impl FnMut(&Value, &Value),
) -> Result<Value, RuntimeError> {
    match admit(&obj, &field, caller)? {
        Store::Dependent => {
            let outputs = match context {
                Some(context) => {
                    crate::object::dispatch::call_object_property_setter_with_outputs_in_context(
                        context,
                        Value::Object(obj),
                        &field,
                        rhs,
                        1,
                    )
                    .await
                }
                None => {
                    call_object_property_setter_with_outputs(Value::Object(obj), &field, rhs, 1)
                        .await
                }
            }?;
            require_single_sequence(outputs)
        }
        Store::Stored => {
            if let Some(old) = obj.properties.get(&field) {
                on_write(old, &rhs);
            }
            obj.properties.insert(field, rhs);
            Ok(Value::Object(obj))
        }
    }
}

fn invalid_handle() -> RuntimeError {
    mex(
        "RunMat:setfield:InvalidHandle",
        "Invalid or deleted handle object.",
    )
}

fn target_object<'a>(
    handle: &HandleRef,
    value: &'a Value,
) -> Result<&'a ObjectInstance, RuntimeError> {
    match value {
        Value::Object(obj)
            if handle.valid
                && obj.class_name == handle.class_name
                && crate::object_handle_flag_valid(obj) =>
        {
            Ok(obj)
        }
        _ => Err(invalid_handle()),
    }
}

pub(super) async fn store_handle(
    context: Option<&crate::context::RuntimeContext>,
    handle: HandleRef,
    field: String,
    rhs: Value,
    caller: Option<&str>,
) -> Result<Value, RuntimeError> {
    if handle
        .class_name
        .is(crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS)
    {
        // Metadata assignment also edits the owning object's dynamic-property
        // definition. Never hold a mutable GC borrow of the metadata handle
        // while that shared domain owner enters the target object.
        let mut metadata = runmat_gc::gc_with_value(&handle.target, |target| {
            target_object(&handle, target).cloned()
        })
        .map_err(|_| invalid_handle())??;
        if !crate::builtins::introspection::dynamicprops::metadata_assignment(
            &mut metadata,
            &field,
            rhs,
        )? {
            return Err(invalid_handle());
        }
        let replacement = Value::Object(metadata);
        let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
            target_object(&handle, target)?;
            runmat_gc::gc_record_handle_write(&handle.target, &replacement);
            Ok::<_, RuntimeError>(std::mem::replace(target, replacement))
        })
        .map_err(|_| invalid_handle())??;
        // Destruction can reenter GC and other owners.
        drop(old);
        return Ok(Value::HandleObject(handle));
    }
    let action = runmat_gc::gc_with_value(&handle.target, |target| {
        admit(target_object(&handle, target)?, &field, caller)
    })
    .map_err(|_| invalid_handle())??;
    match action {
        Store::Dependent => {
            // Existing rooted-sequence owner keeps receiver and RHS managed
            // references live through callback suspension and reentrant GC.
            let roots = crate::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::comma_separated(vec![
                    Value::HandleObject(handle.clone()),
                    rhs.clone(),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)?,
            )
            .map_err(|error| {
                crate::build_runtime_error(format!(
                    "cannot root property setter arguments: {error}"
                ))
                .build()
            })?;
            // Handle setters have no replacement output. Reentrant changes are
            // already published by their own stores and must never be replaced
            // with a pre-call snapshot (which would also be NaN-unsafe).
            match context {
                Some(context) => {
                    crate::object::dispatch::call_object_property_setter_with_outputs_in_context(
                        context,
                        Value::HandleObject(handle.clone()),
                        &field,
                        rhs,
                        0,
                    )
                    .await?;
                }
                None => {
                    call_object_property_setter_with_outputs(
                        Value::HandleObject(handle.clone()),
                        &field,
                        rhs,
                        0,
                    )
                    .await?;
                }
            }
            runmat_gc::gc_with_value(&handle.target, |target| {
                target_object(&handle, target).map(|_| ())
            })
            .map_err(|_| invalid_handle())??;
            drop(roots);
        }
        Store::Stored => {
            let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                let obj = target_object(&handle, target)?;
                // Revalidate the live object, rather than publishing a snapshot.
                match admit(obj, &field, caller)? {
                    Store::Stored => {}
                    Store::Dependent => {
                        return Err(mex(
                            "RunMat:PropertyChanged",
                            "Property changed before assignment",
                        ))
                    }
                }
                runmat_gc::gc_record_handle_write(&handle.target, &rhs);
                let Value::Object(obj) = target else {
                    unreachable!("target validated above")
                };
                Ok::<_, RuntimeError>(obj.properties.insert(field, rhs))
            })
            .map_err(|_| invalid_handle())??;
            // Resource destruction may reenter owners/GC; never run it while
            // holding the managed target's mutable borrow.
            drop(old);
        }
    }
    Ok(Value::HandleObject(handle))
}
