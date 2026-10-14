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
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) async fn addprop_builtin(target: Value, property_name: String) -> BuiltinResult<Value> {
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
    let _receiver_root = runmat_gc::gc_root(handle.target).map_err(|source| {
        dynamic_error_with_source(
            "addprop",
            &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
            "addprop: failed to retain target through metadata allocation",
            source,
        )
    })?;
    let metadata_handle = allocate_dynamic_property_handle(&definition, target.clone())?;
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
        object.insert_dynamic_property(property_name.clone(), published);
        object
            .properties
            .entry(property_name)
            .or_insert_with(empty_default_value);
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
    Ok(Value::HandleObject(metadata_handle))
}

fn admitted_addprop_definition(
    object: &ObjectInstance,
    property_name: &str,
) -> BuiltinResult<DynamicPropertyDef> {
    let selected = crate::class_registry::checked_receiver_projection(object)?;
    if let Some(projection) = &selected {
        if !projection.class_is_or_subclass(&object.class_name, &super::DYNAMICPROPS_CLASS.owned())
        {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_INVALID_TARGET,
                "addprop: selected target does not inherit dynamicprops",
            ));
        }
    } else {
        require_dynamicprops_target(&object.class_name)?;
    }
    let declared = selected.as_ref().map_or_else(
        || {
            crate::class_registry::lookup_property(
                &object.class_name,
                &runmat_types::MemberName::from(property_name),
            )
            .is_some()
        },
        |projection| {
            projection
                .receiver_property(
                    &object.class_name,
                    &runmat_types::MemberName::from(property_name),
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
    Ok(DynamicPropertyDef::new(
        property_name.to_string(),
        object.class_name.clone(),
    ))
}

#[runtime_builtin(
    name = "findprop",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) fn findprop_builtin(target: Value, property_name: String) -> BuiltinResult<Value> {
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
                    return Ok(dynamic_property_handle_from_gc(metadata_handle));
                }
                return Ok(Value::HandleObject(allocate_dynamic_property_handle(
                    def,
                    Value::HandleObject(handle.clone()),
                )?));
            }
            let property = selected.as_ref().map_or_else(
                || {
                    crate::class_registry::lookup_property(
                        &obj.class_name,
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
                        &obj.class_name,
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                },
            );
            if let Some((prop, owner)) = property {
                let def = dynamic_def_from_class_property(
                    &owner,
                    &runmat_types::MemberName::from(property_name.as_str()),
                    &prop,
                );
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                )));
            }
            Ok(empty_metadata_result())
        }
        Value::Object(obj) => {
            let selected = crate::class_registry::checked_receiver_projection(&obj)?;
            if let Some(def) = obj.dynamic_property(&property_name) {
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    def,
                    None,
                    DYNAMIC_PROPERTY_CLASS,
                )));
            }
            let property = selected.as_ref().map_or_else(
                || {
                    crate::class_registry::lookup_property(
                        &obj.class_name,
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
                        &obj.class_name,
                        &runmat_types::MemberName::from(property_name.as_str()),
                    )
                },
            );
            if let Some((prop, owner)) = property {
                let def = dynamic_def_from_class_property(
                    &owner,
                    &runmat_types::MemberName::from(property_name.as_str()),
                    &prop,
                );
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                )));
            }
            Ok(empty_metadata_result())
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
    builtin_path = "crate::builtins::introspection::dynamicprops::operations"
)]
pub(crate) async fn dynamic_property_delete_builtin(prop: Value) -> BuiltinResult<Value> {
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
    delete_metadata_property(&prop_handle, || Ok(()))?;
    Ok(Value::Num(0.0))
}
