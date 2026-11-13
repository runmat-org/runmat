use super::lifecycle::delete_metadata_property;
use super::metadata::*;
use super::{DYNAMIC_PROPERTY_CLASS, STATIC_PROPERTY_METADATA_CLASS};
use crate::BuiltinResult;
use runmat_builtins::{
    DYNAMIC_ERROR_DUPLICATE, DYNAMIC_ERROR_INVALID_METADATA, DYNAMIC_ERROR_INVALID_TARGET,
};
use runmat_macros::runtime_builtin;
use runmat_value::{DynamicPropertyDef, ObjectInstance, Value};

#[runtime_builtin(
    name = "addprop",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) async fn addprop_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    target: Value,
    property_name: runmat_value::Utf8Text,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ADDPROP_DESCRIPTOR,
        &runmat_builtins::ADDPROP_ERROR_OUTPUTS,
        "addprop",
    )?;
    validate_property_name(&property_name, "addprop")?;
    let Value::HandleObject(handle) = &target else {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: target must be a handle object",
        ));
    };
    if !crate::is_handle_valid(handle) {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: target handle is invalid",
        ));
    }
    // Complete class and duplicate admission before any GC allocation. Keep
    // the receiver rooted through metadata allocation, then publish both the
    // property definition and its canonical metadata handle together.
    let definition = runmat_gc::gc_with_value(&handle.target, |value| -> BuiltinResult<_> {
        let Value::Object(object) = value else {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_INVALID_TARGET,
                "addprop: handle target is not an object",
            ));
        };
        admitted_addprop_definition(object, &property_name)
    })
    .map_err(|source| {
        dynamic_error_with_source(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: invalid handle target",
            source,
        )
    })??;
    let _receiver_root =
        crate::execution::RootedValue::in_optional_context(target.clone(), runtime.as_ref())
            .map_err(|source| {
                dynamic_error_with_source(
                    "addprop",
                    &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
                    "addprop: failed to retain target through metadata allocation",
                    source,
                )
            })?;
    let metadata = allocate_dynamic_property_handle(runtime.as_ref(), &definition, target.clone())?;
    let Value::HandleObject(metadata_handle) = metadata.value() else {
        unreachable!("dynamic metadata handle")
    };
    let mut published = definition;
    published.metadata_handle = Some(metadata_handle.target);
    runmat_gc::gc_record_handle_write(
        &handle.target,
        &Value::HandleObject(metadata_handle.clone()),
    );
    runmat_gc::gc_with_value_mut(&handle.target, |value| -> BuiltinResult<()> {
        let Value::Object(object) = value else {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_INVALID_TARGET,
                "addprop: handle target is not an object",
            ));
        };
        // Recheck after allocation in case a GC hook changed observable
        // state. No property or value is published on this rejection path.
        admitted_addprop_definition(object, &property_name)?;
        let default = super::projection::empty_default(runtime.as_ref())?;
        crate::execution::object::replacement::add_dynamic_property(
            runtime.as_ref(),
            object,
            published,
            default.value().clone(),
            |message, source| {
                dynamic_error_with_source(
                    "addprop",
                    &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
                    message,
                    source,
                )
            },
            |message| dynamic_error("addprop", &DYNAMIC_ERROR_INVALID_TARGET, message),
        )?;
        Ok(())
    })
    .map_err(|source| {
        dynamic_error_with_source(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: invalid handle target during publication",
            source,
        )
    })??;
    Ok(metadata)
}

fn admitted_addprop_definition(
    object: &ObjectInstance,
    property_name: &runmat_value::Utf8Text,
) -> BuiltinResult<DynamicPropertyDef> {
    let selected = crate::class_registry::checked_receiver_projection(object)?;
    if let Some(projection) = &selected {
        if !projection.class_is_or_subclass(object.class_name(), &super::DYNAMICPROPS_CLASS.owned())
        {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_INVALID_TARGET,
                "addprop: selected target does not inherit dynamicprops",
            ));
        }
    } else {
        require_dynamicprops_target(object.class_name())?;
    }
    let declared = selected.as_ref().map_or_else(
        || {
            crate::class_registry::lookup_property(
                object.class_name(),
                &runmat_types::MemberName::from(property_name.as_str()),
            )
            .is_some()
        },
        |projection| {
            projection
                .receiver_property(
                    object.class_name(),
                    &runmat_types::MemberName::from(property_name.as_str()),
                )
                .is_some()
        },
    );
    if declared || object.has_dynamic_property(property_name) {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_DUPLICATE,
            format!("addprop: property '{property_name}' already exists"),
        ));
    }
    let mut definition = DynamicPropertyDef::new(String::new(), object.class_name_owner().clone());
    definition.name = property_name.clone();
    Ok(definition)
}

#[runtime_builtin(
    name = "findprop",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) fn findprop_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    target: Value,
    property_name: runmat_value::Utf8Text,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::FINDPROP_DESCRIPTOR,
        &runmat_builtins::FINDPROP_ERROR_OUTPUTS,
        "findprop",
    )?;
    validate_property_name(&property_name, "findprop")?;
    match target {
        Value::HandleObject(handle) => {
            if !crate::is_handle_valid(&handle) {
                return Err(dynamic_error(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    "findprop: target handle is invalid",
                ));
            }
            let target_value = runmat_gc::gc_clone_value(&handle.target).map_err(|source| {
                dynamic_error_with_source(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    "findprop: invalid handle target",
                    source,
                )
            })?;
            let Value::Object(obj) = target_value else {
                return Err(dynamic_error(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    "findprop: handle target is not an object",
                ));
            };
            let selected = crate::class_registry::checked_receiver_projection(&obj)?;
            if let Some(def) = obj.dynamic_property(&property_name) {
                if let Some(metadata_handle) = def.metadata_handle {
                    return dynamic_property_handle_from_gc(runtime, metadata_handle);
                }
                return allocate_dynamic_property_handle(
                    runtime,
                    def,
                    Value::HandleObject(handle.clone()),
                );
            }
            let property = selected.as_ref().map_or_else(
                || {
                    crate::class_registry::lookup_property(
                        obj.class_name(),
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                    .map(|(property, owner)| {
                        (
                            crate::class_registry::receiver_property_declaration(property),
                            owner,
                        )
                    })
                },
                |projection| {
                    projection.receiver_property(
                        obj.class_name(),
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                },
            );
            if let Some((prop, owner)) = property {
                let def = dynamic_def_from_class_property(runtime, &owner, &property_name, &prop)?;
                return super::projection::metadata_object(
                    runtime,
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                );
            }
            super::projection::empty_default(runtime)
        }
        Value::Object(obj) => {
            let selected = crate::class_registry::checked_receiver_projection(&obj)?;
            if let Some(def) = obj.dynamic_property(&property_name) {
                return super::projection::metadata_object(
                    runtime,
                    def,
                    None,
                    DYNAMIC_PROPERTY_CLASS,
                );
            }
            let property = selected.as_ref().map_or_else(
                || {
                    crate::class_registry::lookup_property(
                        obj.class_name(),
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                    .map(|(property, owner)| {
                        (
                            crate::class_registry::receiver_property_declaration(property),
                            owner,
                        )
                    })
                },
                |projection| {
                    projection.receiver_property(
                        obj.class_name(),
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                },
            );
            if let Some((prop, owner)) = property {
                let def = dynamic_def_from_class_property(runtime, &owner, &property_name, &prop)?;
                return super::projection::metadata_object(
                    runtime,
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                );
            }
            super::projection::empty_default(runtime)
        }
        other => Err(dynamic_error(
            "findprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("findprop: target must be object or handle object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "matlab.metadata.DynamicProperty.delete",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) async fn dynamic_property_delete_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    prop: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DYNAMIC_PROPERTY_DELETE_DESCRIPTOR,
        &runmat_builtins::DYNAMIC_PROPERTY_DELETE_ERROR_OUTPUTS,
        "matlab.metadata.DynamicProperty.delete",
    )?;
    let Value::HandleObject(prop_handle) = prop else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property delete requires a metadata handle",
        ));
    };
    if !prop_handle.class_name.is(DYNAMIC_PROPERTY_CLASS) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property delete requires matlab.metadata.DynamicProperty",
        ));
    }
    delete_metadata_property(runtime.as_ref(), &prop_handle, || Ok(()))?;
    Ok(Value::Num(0.0))
}
