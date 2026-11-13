//! Atomic multi-property snapshots retaining original class/native owners.
use super::*;
use crate::context::RuntimeContext;

/// The caller keeps both receiver and update Values independently rooted
/// through this synchronous operation. Replay never collects a second Value or
/// reference array. Duplicate update names retain their last supplied value.
pub(crate) fn replace_properties<'a, I>(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    updates: I,
    allocation_error: impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation_error: impl Fn(String) -> crate::RuntimeError,
) -> Result<(), crate::RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(runtime).map_err(|source| {
        allocation_error("object snapshot authority is no longer current", source)
    })?;
    let policy = HostAllocationReceipt::policy(runtime).map_err(|source| {
        allocation_error("object snapshot policy is no longer current", source)
    })?;
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(allocation_error(
            "object snapshot was cancelled before preparation",
            HostAllocationError::Cancelled,
        ));
    }
    if updates.clone().next().is_none() {
        return Ok(());
    }
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime.expect("managed snapshot has its selected context");
        // Only first occurrences absent from the source need a new admitted
        // key owner; replay avoids an unclaimed deduplication set.
        let new_keys = updates
            .clone()
            .enumerate()
            .filter_map(|(index, (name, _))| {
                (object.property(name).is_none()
                    && !updates.clone().take(index).any(|(prior, _)| prior == name))
                .then_some(name)
            });
        let added = new_keys
            .clone()
            .try_fold(0usize, |count, _| count.checked_add(1))
            .ok_or_else(|| {
                allocation_error(
                    "object snapshot key count overflow",
                    HostAllocationError::Overflow,
                )
            })?;
        let count = object.property_count().checked_add(added).ok_or_else(|| {
            allocation_error(
                "object snapshot property count overflow",
                HostAllocationError::Overflow,
            )
        })?;
        let shape = preparation::Shape {
            properties: count,
            dynamic: object.dynamic_property_count(),
            omission: preparation::Omission::None,
        };
        let receipt = preparation::admit(runtime, object, &shape, new_keys, &allocation_error)?;
        let mut replacement = preparation::prepare(
            runtime,
            object,
            &shape,
            &receipt,
            &allocation_error,
            &presentation_error,
        )?;
        let permit = receipt.output_permit(runtime).map_err(|source| {
            allocation_error("object snapshot permit is no longer current", source)
        })?;
        for (index, (name, value)) in updates.clone().enumerate() {
            receipt.checkpoint(runtime).map_err(|source| {
                allocation_error("object snapshot was cancelled during preparation", source)
            })?;
            if object.property(name).is_some()
                || updates.clone().take(index).any(|(prior, _)| prior == name)
            {
                replacement
                    .try_replace_existing_property(name, value.clone())
                    .map_err(|_| {
                        presentation_error("object snapshot lost an existing property".to_owned())
                    })?;
            } else {
                let key = Utf8Text::try_copy_str_exact_with(name, &permit, |_| {
                    receipt.checkpoint(runtime)
                })
                .map_err(|source| match source {
                    runmat_value::ExactArrayInitializationError::Allocation(source) => {
                        allocation_error("object snapshot key allocation failed", source.into())
                    }
                    runmat_value::ExactArrayInitializationError::Element(source) => {
                        allocation_error("object snapshot key preparation was cancelled", source)
                    }
                })?;
                replacement
                    .try_insert_prepared(key, value.clone())
                    .map_err(|source| {
                        presentation_error(format!(
                            "object snapshot key insertion failed: {}",
                            source.into_parts().0
                        ))
                    })?;
            }
        }
        preparation::publish(
            runtime,
            object,
            replacement,
            &receipt,
            &owner,
            &allocation_error,
        )
    } else {
        if !object.permits_unbounded_edit() {
            return Err(presentation_error(
                "admitted object generation cannot enter an unbounded snapshot edit".to_owned(),
            ));
        }
        let mut replacement = object
            .clone()
            .into_unbounded_builder()
            .expect("ordinary copy eligibility was checked");
        for (name, value) in updates {
            replacement.insert(name.to_owned(), value.clone());
        }
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(allocation_error(
                "object snapshot was cancelled before publication",
                HostAllocationError::Cancelled,
            ));
        }
        if HostAllocationReceipt::policy(runtime).map_err(|source| {
            allocation_error("object snapshot policy changed before publication", source)
        })? != HostAllocationPolicy::Unbounded
        {
            return Err(allocation_error(
                "object snapshot authority changed before publication",
                HostAllocationError::StaleAuthority,
            ));
        }
        *object = replacement.finish();
        Ok(())
    }
}
