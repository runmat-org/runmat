//! Owner-checked dynamic-property deletion and metadata alias lifecycle.
use super::metadata::{dynamic_error, dynamic_error_with_source, metadata_target_and_name};
use super::{DYNAMIC_PROPERTY_CLASS, TARGET_FIELD, VALID_FIELD};
use crate::BuiltinResult;
use runmat_builtins::{DYNAMIC_ERROR_INVALID_METADATA, DYNAMIC_ERROR_INVALID_TARGET};
use runmat_value::{HandleRef, ObjectInstance, Value};

fn owned_definition(
    object: &ObjectInstance,
    name: &str,
    metadata_handle: runmat_gc::GcHandle,
) -> BuiltinResult<()> {
    if object
        .dynamic_property(name)
        .and_then(|definition| definition.metadata_handle)
        != Some(metadata_handle)
    {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata is stale or no longer owns this property",
        ));
    }
    Ok(())
}

fn verify_removal(
    target: &HandleRef,
    name: &str,
    metadata_handle: runmat_gc::GcHandle,
) -> BuiltinResult<()> {
    if !crate::is_handle_valid(target) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            "dynamic property target handle is invalid",
        ));
    }
    runmat_gc::gc_with_value(&target.target, |value| -> BuiltinResult<()> {
        let Value::Object(object) = value else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_TARGET,
                "dynamic property target is not an object",
            ));
        };
        owned_definition(object, name, metadata_handle)
    })
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            "dynamic property target is invalid",
            source,
        )
    })?
}

fn remove_owned_definition(
    runtime: Option<&crate::context::RuntimeContext>,
    target: &HandleRef,
    name: &str,
    metadata_handle: runmat_gc::GcHandle,
) -> BuiltinResult<()> {
    runmat_gc::gc_with_value_mut(&target.target, |value| -> BuiltinResult<()> {
        let Value::Object(object) = value else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_TARGET,
                "dynamic property target is not an object",
            ));
        };
        owned_definition(object, name, metadata_handle)?;
        let (removed, definition) = crate::execution::object::replacement::remove_dynamic_property(
            runtime,
            object,
            name,
            |message, source| {
                dynamic_error_with_source(
                    DYNAMIC_PROPERTY_CLASS,
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    message,
                    source,
                )
            },
            |message| {
                dynamic_error(
                    DYNAMIC_PROPERTY_CLASS,
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    message,
                )
            },
        )?;
        drop(removed);
        drop(definition);
        Ok(())
    })
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            "dynamic property target is invalid during removal",
            source,
        )
    })?
}

fn set_metadata_validity(
    runtime: Option<&crate::context::RuntimeContext>,
    prop_handle: &HandleRef,
    target: &HandleRef,
    valid: bool,
) -> BuiltinResult<()> {
    let allocation = |message: &'static str, source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:DynamicProperty:Metadata",
            message,
            source,
        )
    };
    let owner = crate::context::host_allocation::HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| allocation("dynamic metadata edit authority is unavailable", source))?;
    runmat_gc::gc_with_value_mut(&prop_handle.target, |value| -> BuiltinResult<()> {
        let Value::Object(metadata) = value else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                "dynamic property metadata target is not an object",
            ));
        };
        // Both flags and the target edge publish together. A private copy can
        // fail or be cancelled without exposing an intermediate generation.
        let mut replacement = metadata.clone();
        let fields = [
            (VALID_FIELD, Value::Bool(valid)),
            (crate::HANDLE_VALID_FLAG_PROPERTY, Value::Bool(valid)),
        ];
        let target_value = Value::HandleObject(target.clone());
        crate::execution::object::replacement::replace_properties(
            runtime,
            &mut replacement,
            fields
                .iter()
                .map(|(name, value)| (*name, value))
                .chain(valid.then_some((TARGET_FIELD, &target_value))),
            allocation,
            |message| {
                dynamic_error(
                    DYNAMIC_PROPERTY_CLASS,
                    &DYNAMIC_ERROR_INVALID_METADATA,
                    message,
                )
            },
        )?;
        if !valid && replacement.property(TARGET_FIELD).is_some() {
            crate::execution::object::replacement::remove_property(
                runtime,
                &mut replacement,
                TARGET_FIELD,
                allocation,
                |message| {
                    dynamic_error(
                        DYNAMIC_PROPERTY_CLASS,
                        &DYNAMIC_ERROR_INVALID_METADATA,
                        message,
                    )
                },
            )?;
        }
        if let Some(owner) = &owner {
            owner
                .check(runtime.expect("managed metadata owner has context"))
                .map_err(|source| allocation("dynamic metadata edit authority changed", source))?;
        }
        if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
            return Err(allocation(
                "dynamic metadata edit was cancelled",
                crate::context::host_allocation::HostAllocationError::Cancelled,
            ));
        }
        if valid {
            runmat_gc::gc_record_handle_write(&prop_handle.target, &target_value);
        }
        *metadata = replacement;
        Ok(())
    })
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata handle is invalid during deletion",
            source,
        )
    })?
}

pub(super) fn delete_metadata_property(
    runtime: Option<&crate::context::RuntimeContext>,
    prop_handle: &HandleRef,
    after_invalidation: impl FnOnce() -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let _metadata_root = crate::execution::RootedValue::in_optional_context(
        Value::HandleObject(prop_handle.clone()),
        runtime,
    )
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata handle cannot be retained",
            source,
        )
    })?;
    let metadata = runmat_gc::gc_clone_value(&prop_handle.target).map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata handle is invalid",
            source,
        )
    })?;
    let Value::Object(metadata) = metadata else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata target is not an object",
        ));
    };
    let (target, name) = metadata_target_and_name(runtime, &metadata)?;
    // The actual metadata handle is already rooted. Retire this snapshot
    // before creating the separate owner root needed after edge invalidation.
    drop(metadata);
    let _target_root = crate::execution::RootedValue::in_optional_context(
        Value::HandleObject(target.clone()),
        runtime,
    )
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            "dynamic property owner cannot be retained",
            source,
        )
    })?;
    verify_removal(&target, &name, prop_handle.target)?;
    // Invalidate all aliases before removing the owning definition. A fault
    // after this point cannot leave a still-valid alias to a missing property.
    set_metadata_validity(runtime, prop_handle, &target, false)?;
    let removal = after_invalidation()
        .and_then(|()| remove_owned_definition(runtime, &target, &name, prop_handle.target));
    if let Err(mut failure) = removal {
        // No property was removed on either rejection path. Restore metadata
        // before returning the original failure when the GC permits it. If
        // restoration itself fails, preserve the original identifier/source
        // and report the second failure without claiming an atomic rollback.
        if let Err(rollback) = set_metadata_validity(runtime, prop_handle, &target, true) {
            failure.message = format!(
                "{}; metadata restoration also failed: {}",
                failure.message, rollback.message
            );
        }
        return Err(failure);
    }
    Ok(())
}
