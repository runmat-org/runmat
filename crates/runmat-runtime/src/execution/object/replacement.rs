//! Original-generation atomic Object property replacement. Callers supply only
//! diagnostic presentation; allocation and publication have one canonical owner.
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_value::{
    ExactObjectBuilder, ObjectEditPreparationError, ObjectInstance, Utf8Text, Value,
};

mod preparation;
mod batch;
mod dynamic;
pub(crate) use batch::replace_properties;
pub(crate) use dynamic::{add_dynamic_property, replace_dynamic_definition};
mod copy;
pub(crate) use copy::copy_generation;

pub(crate) fn replace_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &mut ObjectInstance,
    name: &str,
    value: Value,
    allocation_error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation_error: impl Fn(String) -> crate::RuntimeError,
) -> Result<Option<Value>, crate::RuntimeError> {
    edit_property(
        runtime,
        obj,
        name,
        Some(value),
        false,
        allocation_error,
        presentation_error,
    )
}

/// Remove one declared slot without changing class, dynamic declarations, or
/// native/selected owners. The source remains published until its complete
/// private destination passes the final checkpoint.
pub(crate) fn remove_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &mut ObjectInstance,
    name: &str,
    allocation_error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation_error: impl Fn(String) -> crate::RuntimeError,
) -> Result<Value, crate::RuntimeError> {
    if obj.property(name).is_none() {
        return Err(presentation_error(format!(
            "object has no property `{name}`"
        )));
    }
    edit_property(
        runtime,
        obj,
        name,
        None,
        false,
        allocation_error,
        presentation_error,
    )
    .map(|old| old.expect("existing property was admitted before removal"))
}

pub(crate) fn remove_dynamic_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &mut ObjectInstance,
    name: &str,
    allocation_error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation_error: impl Fn(String) -> crate::RuntimeError,
) -> Result<(Value, runmat_value::DynamicPropertyDef), crate::RuntimeError> {
    let definition = obj
        .dynamic_property(name)
        .cloned()
        .ok_or_else(|| presentation_error(format!("object has no dynamic property `{name}`")))?;
    if obj.property(name).is_none() {
        return Err(presentation_error(format!(
            "dynamic property `{name}` has no value"
        )));
    }
    let old = edit_property(
        runtime,
        obj,
        name,
        None,
        true,
        allocation_error,
        presentation_error,
    )?;
    Ok((
        old.expect("dynamic value was admitted before removal"),
        definition,
    ))
}

fn edit_property(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: &mut ObjectInstance,
    name: &str,
    value: Option<Value>,
    remove_dynamic: bool,
    allocation_error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation_error: impl Fn(String) -> crate::RuntimeError,
) -> Result<Option<Value>, crate::RuntimeError> {
    let owner = HostAllocationReceipt::capture_owner(runtime).map_err(|source| {
        allocation_error(
            "object property edit authority is no longer current",
            source,
        )
    })?;
    let policy = HostAllocationReceipt::policy(runtime).map_err(|source| {
        allocation_error("object property edit policy is no longer current", source)
    })?;
    if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
        return Err(allocation_error(
            "object property edit was cancelled before mutation",
            HostAllocationError::Cancelled,
        ));
    }
    let value = if let Some(value) = value {
        let value = match obj.try_replace_property_unique(name, value) {
            Ok(old) => return Ok(Some(old)),
            Err(value) => value,
        };
        Some(value)
    } else {
        None
    };
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime.expect("Managed policy has a selected runtime");
        let removing = value.is_none();
        let inserting = !removing && obj.property(name).is_none();
        let property_capacity = obj
            .property_count()
            .checked_add(usize::from(inserting))
            .and_then(|count| count.checked_sub(usize::from(removing)))
            .ok_or_else(|| {
                allocation_error(
                    "object property capacity overflow",
                    HostAllocationError::Overflow,
                )
            })?;
        let dynamic_capacity = obj.dynamic_property_count() - usize::from(remove_dynamic);
        let shape = preparation::Shape {
            properties: property_capacity,
            dynamic: dynamic_capacity,
            omission: if remove_dynamic {
                preparation::Omission::Dynamic(name)
            } else if removing {
                preparation::Omission::Property(name)
            } else {
                preparation::Omission::None
            },
        };
        let receipt = preparation::admit(
            runtime,
            obj,
            &shape,
            inserting.then_some(name).into_iter(),
            &allocation_error,
        )?;
        let mut replacement = preparation::prepare(
            runtime,
            obj,
            &shape,
            &receipt,
            &allocation_error,
            &presentation_error,
        )?;
        let permit = receipt.output_permit(runtime).map_err(|source| {
            allocation_error("object replacement permit is no longer current", source)
        })?;
        let old = if let Some(value) = value {
            if inserting {
                let key = Utf8Text::try_copy_str_exact_with(name, &permit, |_| {
                    Ok::<(), std::convert::Infallible>(())
                })
                .map_err(|source| match source {
                    runmat_value::ExactArrayInitializationError::Allocation(source) => {
                        allocation_error(
                            "object property key allocation failed",
                            HostAllocationError::Numeric(source),
                        )
                    }
                    runmat_value::ExactArrayInitializationError::Element(never) => match never {},
                })?;
                replacement
                    .try_insert_prepared(key, value)
                    .map_err(|error| {
                        let (kind, _key, _value) = error.into_parts();
                        presentation_error(format!(
                            "object replacement key insertion failed: {kind:?}"
                        ))
                    })?
            } else {
                Some(
                    replacement
                        .try_replace_existing_property(name, value)
                        .map_err(|_| {
                            presentation_error(
                                "object replacement lost an existing property".to_owned(),
                            )
                        })?,
                )
            }
        } else {
            obj.property(name).cloned()
        };
        preparation::publish(
            runtime,
            obj,
            replacement,
            &receipt,
            &owner,
            &allocation_error,
        )?;
        return Ok(old);
    }
    if !obj.permits_unbounded_edit() {
        return Err(presentation_error(
            "admitted object generation cannot enter an unbounded property edit".to_owned(),
        ));
    }
    let mut builder = obj
        .clone()
        .into_unbounded_builder()
        .expect("ordinary copy eligibility was checked");
    let old = if let Some(value) = value {
        builder.insert(name.to_owned(), value)
    } else {
        let old = builder.remove(name);
        if remove_dynamic {
            builder.remove_dynamic(name);
        }
        old
    };
    if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
        return Err(allocation_error(
            "object property edit was cancelled before publication",
            HostAllocationError::Cancelled,
        ));
    }
    if HostAllocationReceipt::policy(runtime).map_err(|source| {
        allocation_error(
            "object property edit policy changed before publication",
            source,
        )
    })? != HostAllocationPolicy::Unbounded
    {
        return Err(allocation_error(
            "object property edit authority changed before publication",
            HostAllocationError::StaleAuthority,
        ));
    }
    *obj = builder.finish();
    Ok(old)
}
