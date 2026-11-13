//! Publish one declaration and its value in the same original Object generation.
use super::*;
use crate::context::RuntimeContext;
use runmat_value::DynamicPropertyDef;

/// The caller roots the source, new value and declaration's metadata handle
/// throughout preparation. Refusal never changes the published receiver.
pub(crate) fn add_dynamic_property(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    definition: DynamicPropertyDef,
    value: Value,
    error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation: impl Fn(String) -> crate::RuntimeError,
) -> Result<(), crate::RuntimeError> {
    write_dynamic_property(
        runtime,
        object,
        definition,
        Edit::Add(value),
        error,
        presentation,
    )
}

pub(crate) fn replace_dynamic_definition(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    definition: DynamicPropertyDef,
    error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation: impl Fn(String) -> crate::RuntimeError,
) -> Result<(), crate::RuntimeError> {
    write_dynamic_property(
        runtime,
        object,
        definition,
        Edit::Replace,
        error,
        presentation,
    )
}

enum Edit {
    Add(Value),
    Replace,
}

fn write_dynamic_property(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    definition: DynamicPropertyDef,
    edit: Edit,
    error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation: impl Fn(String) -> crate::RuntimeError,
) -> Result<(), crate::RuntimeError> {
    let additional = usize::from(matches!(&edit, Edit::Add(_)));
    if matches!(&edit, Edit::Replace) && !object.has_dynamic_property(&definition.name) {
        return Err(presentation(format!(
            "dynamic property `{}` no longer exists",
            definition.name
        )));
    }
    if matches!(&edit, Edit::Add(_))
        && (object.has_dynamic_property(&definition.name)
            || object.property(&definition.name).is_some())
    {
        return Err(presentation(format!(
            "dynamic property `{}` already exists",
            definition.name
        )));
    }
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| error("dynamic declaration authority is unavailable", source))?;
    let policy = HostAllocationReceipt::policy(runtime)
        .map_err(|source| error("dynamic declaration policy is unavailable", source))?;
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(error(
            "dynamic declaration was cancelled",
            HostAllocationError::Cancelled,
        ));
    }
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime.expect("managed dynamic declaration has context");
        let overflow = || {
            error(
                "dynamic declaration geometry overflow",
                HostAllocationError::Overflow,
            )
        };
        let shape = preparation::Shape {
            properties: object
                .property_count()
                .checked_add(additional)
                .ok_or_else(overflow)?,
            dynamic: object
                .dynamic_property_count()
                .checked_add(additional)
                .ok_or_else(overflow)?,
            omission: preparation::Omission::None,
        };
        let mut extra = preparation::ExtraMetadata::default();
        for text in [&definition.name, &definition.description] {
            if !text.has_retained_claim() && text.capacity() != 0 {
                let bytes =
                    Utf8Text::requested_bytes_for_capacity(text.capacity()).map_err(|source| {
                        error(
                            "dynamic declaration text geometry is invalid",
                            source.into(),
                        )
                    })?;
                extra.requested_bytes = extra
                    .requested_bytes
                    .checked_add(bytes)
                    .ok_or_else(overflow)?;
                extra.retained_claims += 1;
            }
        }
        if !definition.defining_class.has_retained_claim()
            && !definition
                .defining_class
                .shares_allocation_with(object.class_name_owner())
        {
            let bytes = definition
                .defining_class
                .requested_bytes()
                .map_err(|source| {
                    error(
                        "dynamic declaration class geometry is invalid",
                        source.into(),
                    )
                })?;
            extra.requested_bytes = extra
                .requested_bytes
                .checked_add(bytes)
                .ok_or_else(overflow)?;
            extra.retained_claims += 1;
        }
        let receipt = preparation::admit_with_metadata(
            runtime,
            object,
            &shape,
            std::iter::empty(),
            extra,
            &error,
        )?;
        let mut output =
            preparation::prepare(runtime, object, &shape, &receipt, &error, &presentation)?;
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| error("dynamic declaration permit is unavailable", source))?;
        let definition = definition.try_admit_exact(&permit).map_err(|source| {
            error(
                "dynamic declaration metadata admission failed",
                source.source().into(),
            )
        })?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| error("dynamic declaration preparation was cancelled", source))?;
        if let Edit::Add(value) = &edit {
            output
                .try_insert_prepared(definition.name.clone(), value.clone())
                .map_err(|source| {
                    presentation(format!(
                        "dynamic value insertion failed: {}",
                        source.into_parts().0
                    ))
                })?;
        }
        if matches!(&edit, Edit::Replace) {
            let name = definition.name.clone();
            output
                .try_replace_existing_dynamic_property(&name, definition)
                .map_err(|_| {
                    presentation("dynamic declaration disappeared during preparation".into())
                })?;
        } else {
            output
                .try_insert_dynamic_prepared(definition)
                .map_err(|source| {
                    presentation(format!(
                        "dynamic declaration insertion failed: {}",
                        source.into_parts().0
                    ))
                })?;
        }
        preparation::publish(runtime, object, output, &receipt, &owner, &error)
    } else {
        if !object.permits_unbounded_edit() {
            return Err(presentation(
                "admitted object cannot enter an ordinary dynamic declaration edit".into(),
            ));
        }
        let mut output = object
            .clone()
            .into_unbounded_builder()
            .expect("ordinary source eligibility checked");
        output.insert_dynamic(definition.name.as_str().to_owned(), definition.clone());
        if let Edit::Add(value) = edit {
            output.insert_text(definition.name, value);
        }
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(error(
                "dynamic declaration was cancelled before publication",
                HostAllocationError::Cancelled,
            ));
        }
        if let Some(owner) = &owner {
            owner
                .check(runtime.expect("managed owner has context"))
                .map_err(|source| error("dynamic declaration source changed", source))?;
        }
        *object = output.finish();
        Ok(())
    }
}
