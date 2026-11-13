//! Admitted map-only replacement retaining original column/shape generations.
use super::*;
use runmat_value::{
    ExactArrayClaimPromotionError, ExactHostAllocationPermit, ExactHostNumericAllocationError,
    OutputListStorage, SharedShape, StructArrayBuilder, UnboundedStructArrayBuilder,
};

/// The caller holds the source root lease through publication. No Value child
/// is cloned: only immutable column/shape owners are retained. Each original
/// physical owner is promoted separately, so repeated aliases never require
/// a scratch identity set or duplicate retained claims.
pub fn construct_struct_array_from_columns_in_context<'a, I>(
    runtime: &RuntimeContext,
    columns: I,
    shape: &SharedShape,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a OutputListStorage)> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(runtime, source))?;
    let count = columns.clone().try_fold(0usize, |count, _| {
        check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
        count
            .checked_add(1)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))
    })?;
    let managed = HostAllocationReceipt::policy(Some(runtime))
        .map_err(|source| allocation(runtime, source))?
        == HostAllocationPolicy::Managed;
    if managed {
        if let Some(bytes) = shape
            .unclaimed_shared_bytes()
            .map_err(|source| allocation(runtime, source.into()))?
        {
            promote(runtime, bytes, |permit| {
                shape.try_promote_shared_claim(permit)
            })?;
        }
        for (_, values) in columns.clone() {
            check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
            let bytes = values.unclaimed_shared_bytes().map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "StructArrayConstruction",
                    "column is not a canonical shared owner",
                    source,
                )
            })?;
            if let Some(bytes) = bytes {
                promote(runtime, bytes, |permit| {
                    values
                        .try_promote_shared_claim(permit)
                        .map_err(|source| match source {
                            ExactArrayClaimPromotionError::Allocation(source) => source,
                            ExactArrayClaimPromotionError::UnsharedBacking => {
                                unreachable!("canonical shared column was checked")
                            }
                        })
                })?;
            }
        }
    }
    let receipt = admit_struct_generation_for::<OutputListStorage, _>(
        Some(runtime),
        columns.clone().map(|(name, _)| name),
        count,
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
        let mut builder = builder.map_err(|source| array_error(runtime, source))?;
        for (name, values) in columns {
            check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let key = copy_key(runtime, receipt, &permit, name)?;
            builder
                .try_insert_prepared(key, values.clone())
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
            .map_err(|source| array_error(runtime, source))?;
        for (name, values) in columns {
            builder
                .insert(name.to_owned(), values.clone())
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
pub(super) fn array_error(
    runtime: &RuntimeContext,
    source: runmat_value::StructArrayConstructionError,
) -> RuntimeError {
    if let runmat_value::StructArrayConstructionError::Allocation(source) = source {
        return allocation(runtime, source.into());
    }
    crate::runtime_error::semantic_error_with_source(
        "StructArrayConstruction",
        "invalid structure-array destination",
        source,
    )
}
pub(super) fn promote(
    runtime: &RuntimeContext,
    bytes: usize,
    operation: impl FnOnce(
        &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError>,
) -> Result<(), RuntimeError> {
    let bytes =
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
    let output = HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
        .map_err(|source| allocation(runtime, source))?;
    let plan = HostAllocationPlan::new(0, output).map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|source| allocation(runtime, source))?
        .ok_or_else(|| allocation(runtime, HostAllocationError::StaleAuthority))?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let result = operation(&permit);
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))?;
    result.map_err(|source| allocation(runtime, source.into()))
}
