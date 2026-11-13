//! Exact slot-copy destination for structural array edits. Children retain
//! original immutable generations; only the destination Value slots are new.
use super::*;
use runmat_value::{
    ExactArrayInitializationError, OutputListStorage, SharedShape, StructArrayBuilder,
    UnboundedStructArrayBuilder,
};

/// Original source roots must remain held by the caller through publication.
/// This is checkpointed synchronous construction, not an async yield budget.
pub fn construct_struct_array_from_values_in_context<'a, I>(
    runtime: &RuntimeContext,
    columns: I,
    shape: &SharedShape,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a [Value])> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(runtime, source))?;
    let count = columns.clone().try_fold(0usize, |count, _| {
        check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
        count
            .checked_add(1)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))
    })?;
    StructArrayBuilder::quote(count, shape)
        .map_err(|source| super::array::array_error(runtime, source))?;
    let length = shape.iter().copied().product::<usize>(); // checked by the canonical quote
    for (_, values) in columns.clone() {
        check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
        if values.len() != length {
            return Err(changed());
        }
    }
    if owner.is_some() {
        if let Some(bytes) = shape
            .unclaimed_shared_bytes()
            .map_err(|source| allocation(runtime, source.into()))?
        {
            super::array::promote(runtime, bytes, |permit| {
                shape.try_promote_shared_claim(permit)
            })?;
        }
    }
    let bytes = OutputListStorage::requested_bytes(length)
        .map_err(|source| allocation(runtime, source.into()))?
        .checked_mul(count)
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    let claims = usize::from(length != 0)
        .checked_mul(count)
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    let receipt = admit_struct_generation_extra_for::<OutputListStorage, _>(
        Some(runtime),
        columns.clone().map(|(name, _)| name),
        count,
        bytes,
        claims,
    )
    .map_err(|source| allocation(runtime, source))?;
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    let mut emitted = 0usize;
    let array = if let Some(receipt) = &receipt {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let builder = StructArrayBuilder::try_new_retaining_shape(count, shape.clone(), &permit);
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let mut builder = builder.map_err(|source| super::array::array_error(runtime, source))?;
        for (name, source) in columns {
            check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
            if source.len() != length {
                return Err(changed());
            }
            let key = copy_key(runtime, receipt, &permit, name)?;
            let values = OutputListStorage::try_initialize(length, &permit, |index| {
                if index % 4096 == 0 {
                    receipt.checkpoint(runtime)?;
                }
                Ok::<_, HostAllocationError>(source[index].clone())
            });
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let values = values.map_err(|source| {
                allocation(
                    runtime,
                    match source {
                        ExactArrayInitializationError::Allocation(source) => source.into(),
                        ExactArrayInitializationError::Element(source) => source,
                    },
                )
            })?;
            builder
                .try_insert_prepared(key, values)
                .map_err(|_| changed())?;
            emitted = emitted
                .checked_add(1)
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        builder.finish()
    } else {
        let mut builder = UnboundedStructArrayBuilder::with_capacity(shape.clone(), count)
            .map_err(|source| super::array::array_error(runtime, source))?;
        for (name, values) in columns {
            if values.len() != length {
                return Err(changed());
            }
            let values = if values.is_empty() {
                OutputListStorage::empty()
            } else {
                OutputListStorage::shared_unbounded(values.to_vec())
            };
            builder
                .insert(name.to_owned(), values)
                .map_err(|_| changed())?;
            emitted = emitted
                .checked_add(1)
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
        builder.finish()
    };
    if emitted != count {
        return Err(changed());
    }
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    RootedValue::in_context(Value::StructArray(array), runtime)
}
fn changed() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "StructArrayConstruction",
        "column iterator changed or its shape is inconsistent",
    )
}
