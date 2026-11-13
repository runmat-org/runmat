//! Atomic publication of the stored handle-validity flag.
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{RuntimeError, Value, HANDLE_VALID_FLAG_PROPERTY};
use runmat_value::HandleRef;

pub(crate) fn set(
    runtime: Option<&RuntimeContext>,
    handle: &HandleRef,
    valid: bool,
) -> Result<bool, RuntimeError> {
    let _receiver = RootedValue::in_optional_context(Value::HandleObject(handle.clone()), runtime)?;
    let snapshot = runmat_gc::gc_with_value(&handle.target, |value| match value {
        Value::Object(object) if object.class_name() == handle.class_name.as_identity() => {
            Some(object.clone())
        }
        _ => None,
    })
    .map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "HandleValidity",
            "cannot inspect the handle target",
            source,
        )
    })?;
    let Some(snapshot) = snapshot else {
        return Ok(false);
    };
    let mut replacement = snapshot.clone();
    crate::execution::object::replacement::replace_property(
        runtime,
        &mut replacement,
        HANDLE_VALID_FLAG_PROPERTY,
        Value::Bool(valid),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:HandleValidityAllocation",
                message,
                source,
            )
        },
        |message| crate::runtime_error::semantic_error("HandleValidity", message),
    )?;
    let replacement = RootedValue::in_optional_context(Value::Object(replacement), runtime)?;
    // Allocation/root admission can reenter the host. Only the exact inspected
    // generation may be replaced, and no allocation runs under the GC borrow.
    let retired = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(current) = target else {
            return None;
        };
        if !current.shares_generation_with(&snapshot) {
            return None;
        }
        runmat_gc::gc_record_handle_write(&handle.target, replacement.value());
        Some(std::mem::replace(target, replacement.value().clone()))
    })
    .map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "HandleValidity",
            "cannot publish the handle validity flag",
            source,
        )
    })?;
    let published = retired.is_some();
    drop(retired);
    Ok(published)
}
