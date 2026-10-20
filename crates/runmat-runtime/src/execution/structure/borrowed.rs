//! Direct retained-field publication. The caller must retain the original
//! root lease until this helper has returned the newly rooted generation.
use super::*;

pub(crate) fn construct_struct_from_fields_in_optional_context<'name, 'value, I>(
    runtime: Option<&RuntimeContext>,
    fields: I,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'name str, &'value Value)> + Clone,
{
    if let Some(runtime) = runtime {
        return construct_struct_from_fields_in_context(runtime, fields);
    }
    let allocation = |source| {
        host_allocation_runtime_error(
            None,
            "RunMat:Struct:Allocation",
            "structure backing admission failed",
            source,
        )
    };
    let policy = HostAllocationReceipt::policy(None).map_err(allocation)?;
    if policy != HostAllocationPolicy::Unbounded {
        return Err(allocation(HostAllocationError::MissingManagedOwner));
    }
    let mut builder = UnboundedStructValueBuilder::with_capacity(fields.clone().count());
    for (name, value) in fields {
        builder.insert(name, value.clone());
    }
    RootedValue::in_optional_context(Value::Struct(builder.finish()), None)
}

/// Clones only canonical original-generation Value owners into final slots.
/// The complete Value clone invariant is a prerequisite of this boundary;
/// no temporary names or RootedValues carrier is constructed here.
pub fn construct_struct_from_fields_in_context<'name, 'value, I>(
    runtime: &RuntimeContext,
    fields: I,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'name str, &'value Value)> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(runtime, source))?;
    let count = fields.clone().try_fold(0usize, |count, _| {
        check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
        count
            .checked_add(1)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))
    })?;
    let receipt =
        admit_struct_generation_counted(Some(runtime), fields.clone().map(|(name, _)| name), count)
            .map_err(|source| allocation(runtime, source))?;
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    let mut emitted = 0usize;
    let structure = if let Some(receipt) = &receipt {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let builder = StructValueBuilder::try_new(count, &permit);
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let mut builder = builder.map_err(|source| allocation(runtime, source.into()))?;
        for (name, value) in fields {
            check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let key = copy_key(runtime, receipt, &permit, name)?;
            builder
                .try_insert_prepared(key, value.clone())
                .map_err(|source| {
                    if let Err(cause) = receipt.checkpoint(runtime) {
                        return allocation(runtime, cause);
                    }
                    crate::runtime_error::semantic_error_with_source(
                        "StructConstruction",
                        "field iterator changed during admitted publication",
                        source.into_parts().0,
                    )
                })?;
            emitted = emitted
                .checked_add(1)
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        builder.finish()
    } else {
        let mut builder = UnboundedStructValueBuilder::with_capacity(count);
        for (name, value) in fields {
            builder.insert(name, value.clone());
            emitted = emitted
                .checked_add(1)
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
        builder.finish()
    };
    if emitted != count {
        return Err(crate::runtime_error::semantic_error(
            "StructConstruction",
            "field iterator changed during admitted publication",
        ));
    }
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    RootedValue::in_context(Value::Struct(structure), runtime)
}
