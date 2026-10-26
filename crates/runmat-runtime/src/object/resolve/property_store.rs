//! Default property stores after the caller has resolved subsasgn exactly once.
//! Metadata/access admission is shared; value objects return replacements while
//! handle objects publish to their authentic target or invoke an in-place setter.
use super::*;
use runmat_value::{HandleRef, ObjectInstance};

enum Store {
    Stored,
    Dependent,
}

fn admit(
    obj: &ObjectInstance,
    field: &str,
    caller: MemberCaller<'_>,
) -> Result<Store, RuntimeError> {
    let selected = crate::class_registry::checked_receiver_projection(obj)?;
    let property = match &selected {
        Some(projection) => projection.receiver_property(obj.class_name(), &field.into()),
        None => crate::class_registry::lookup_property(obj.class_name(), &field.into()).map(
            |(property, owner)| {
                (
                    crate::class_registry::receiver_property_declaration(property),
                    owner,
                )
            },
        ),
    };
    if let Some((property, owner)) = property {
        if property.is_static {
            return Err(mex(
                "RunMat:PropertyStaticAccess",
                &format!(
                    "Property '{field}' is static; use classref('{}').{field}",
                    obj.class_name()
                ),
            ));
        }
        if property.is_constant {
            return Err(mex(
                IDENT_PROPERTY_READ_ONLY,
                &format!("Property '{field}' is constant"),
            ));
        }
        let permitted = match &selected {
            Some(projection) => projection.receiver_member_accessible(
                &owner,
                property.set_access,
                caller.class().as_ref(),
            ),
            None => access_permitted(&owner, &property.set_access, caller),
        };
        if !permitted {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{field}' is private"),
            ));
        }
        if property.is_dependent {
            let setter = crate::object_property_setter_name(field);
            if selected.as_ref().map_or_else(
                || {
                    crate::class_registry::lookup_method(obj.class_name(), &setter.as_str().into())
                        .is_some()
                },
                |projection| {
                    projection.receiver_declares_method(obj.class_name(), &setter.as_str().into())
                },
            ) {
                return Ok(Store::Dependent);
            }
            return Err(mex(
                IDENT_PROPERTY_READ_ONLY,
                &format!("Dependent property '{field}' has no set method"),
            ));
        }
        return Ok(Store::Stored);
    }
    let caller_class = caller.class();
    let dynamic_access = match &selected {
        Some(projection) => obj.dynamic_property(field).map(|definition| {
            projection.receiver_member_accessible(
                definition.defining_class.as_identity(),
                definition.set_access,
                caller_class.as_ref(),
            )
        }),
        None => dynamicprops::dynamic_property_assignment_access(obj, field, caller_class.as_ref()),
    };
    if let Some(permitted) = dynamic_access {
        if !permitted {
            return Err(mex(
                IDENT_PROPERTY_PRIVATE_ACCESS,
                &format!("Property '{field}' is private"),
            ));
        }
        return Ok(Store::Stored);
    }
    if selected.is_some() || crate::class_registry::get_class(obj.class_name()).is_some() {
        Err(format!(
            "Undefined property '{field}' for class {}",
            obj.class_name()
        )
        .into())
    } else {
        Err(format!("Unknown class {}", obj.class_name()).into())
    }
}

pub(super) async fn store_value(
    context: Option<&crate::context::RuntimeContext>,
    mut obj: ObjectInstance,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller: MemberCaller<'_>,
    on_write: &mut impl FnMut(&Value, &Value),
) -> Result<RootedValue, RuntimeError> {
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
            require_single_rooted_sequence(outputs)
        }
        Store::Stored => {
            if let Some(old) = obj.property(&field) {
                on_write(old, &rhs);
            }
            replace_stored_property(context, &mut obj, &field, rhs)?;
            RootedValue::in_optional_context(Value::Object(obj), context)
        }
    }
}

fn invalid_handle() -> RuntimeError {
    mex(
        "RunMat:setfield:InvalidHandle",
        "Invalid or deleted handle object.",
    )
}

pub(super) fn require_valid_handle(handle: &HandleRef) -> Result<(), RuntimeError> {
    if handle.valid {
        Ok(())
    } else {
        Err(invalid_handle())
    }
}

fn target_object<'a>(
    handle: &HandleRef,
    value: &'a Value,
) -> Result<&'a ObjectInstance, RuntimeError> {
    match value {
        Value::Object(obj)
            if handle.valid
                && obj.class_name() == &handle.class_name
                && crate::object_handle_flag_valid(obj) =>
        {
            Ok(obj)
        }
        _ => Err(invalid_handle()),
    }
}

fn replace_stored_property(
    context: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    field: &str,
    rhs: Value,
) -> Result<Option<Value>, RuntimeError> {
    crate::execution::object::replacement::replace_property(
        context,
        object,
        field,
        rhs,
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                context,
                "RunMat:PropertyAssignment",
                message,
                source,
            )
        },
        |message| mex("RunMat:PropertyAssignment", &message),
    )
}

pub(super) async fn store_handle(
    context: Option<&crate::context::RuntimeContext>,
    handle: HandleRef,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller: MemberCaller<'_>,
) -> Result<RootedValue, RuntimeError> {
    if handle
        .class_name
        .is(crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_CLASS)
    {
        let receiver =
            RootedValue::in_optional_context(Value::HandleObject(handle.clone()), context)?;
        // Metadata assignment also edits the owning object's dynamic-property
        // definition. Never hold a mutable GC borrow of the metadata handle
        // while that shared domain owner enters the target object.
        let mut metadata = runmat_gc::gc_with_value(&handle.target, |target| {
            target_object(&handle, target).cloned()
        })
        .map_err(|_| invalid_handle())??;
        if !crate::builtins::introspection::dynamicprops::metadata_assignment(
            context,
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
        return Ok(receiver);
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
            let result = RootedValue::new(Value::HandleObject(handle))
                .map_err(|error| error.into_runtime_error())?;
            drop(roots);
            return Ok(result);
        }
        Store::Stored => {
            // Prepare outside the GC borrow. Allocation and retirement can
            // reenter owners; publication checks the original generation.
            let snapshot = runmat_gc::gc_with_value(&handle.target, |target| {
                target_object(&handle, target).cloned()
            })
            .map_err(|_| invalid_handle())??;
            let mut replacement = snapshot.clone();
            replace_stored_property(context, &mut replacement, &field, rhs)?;
            let replacement =
                RootedValue::in_optional_context(Value::Object(replacement), context)?;
            let prepared = replacement.into_value_and_lease();
            let _replacement_roots = prepared.1;
            let replacement = prepared.0;
            let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                let live = target_object(&handle, target)?;
                if !live.shares_generation_with(&snapshot)
                    || !matches!(admit(live, &field, caller)?, Store::Stored)
                {
                    return Err(mex(
                        "RunMat:PropertyChanged",
                        "Property changed before assignment",
                    ));
                }
                runmat_gc::gc_record_handle_write(&handle.target, &replacement);
                Ok::<_, RuntimeError>(std::mem::replace(target, replacement))
            })
            .map_err(|_| invalid_handle())??;
            // Resource destruction may reenter owners/GC; never run it while
            // holding the managed target's mutable borrow.
            drop(old);
        }
    }
    RootedValue::in_optional_context(Value::HandleObject(handle), context)
}
