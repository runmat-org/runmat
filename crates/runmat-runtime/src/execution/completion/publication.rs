//! Original collection publication, never a second argument payload batch.
use super::*;
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};

pub(super) fn ordinary(values: OutputListStorage) -> OutputListStorage {
    if values.is_shallow_clone() {
        return values;
    }
    let values = values
        .try_into_owned()
        .expect("unpublished owned collection is unique")
        .try_into_unbounded_vec()
        .expect("unpublished ordinary collection has no claim");
    if values.capacity() == 0 {
        OutputListStorage::empty()
    } else {
        OutputListStorage::shared_unbounded(values)
    }
}
pub(super) fn in_context(
    values: OutputListStorage,
    runtime: &RuntimeContext,
) -> Result<OutputListStorage, crate::RuntimeError> {
    let allocation = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Call:Arguments",
            "original argument collection publication failed",
            source,
        )
    };
    let owner = HostAllocationReceipt::capture_owner(Some(runtime)).map_err(allocation)?;
    let bytes = if values.is_shallow_clone() {
        values
            .unclaimed_shared_bytes()
            .map_err(|source| match source {
                runmat_value::ExactArrayClaimPromotionError::Allocation(source) => {
                    allocation(source.into())
                }
                source => crate::runtime_error::semantic_error_with_source(
                    "RunMat:Call:Arguments",
                    "argument collection is not a canonical shared owner",
                    source,
                ),
            })?
    } else {
        Some(
            OutputListStorage::requested_bytes(values.capacity())
                .map_err(|e| allocation(e.into()))?,
        )
    };
    let Some(bytes) = bytes else {
        return Ok(values);
    };
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(HostAllocationError::Overflow))?,
        usize::from(bytes != 0),
    )
    .map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let result = if let Some(receipt) = &receipt {
        let permit = receipt.output_permit(runtime).map_err(allocation)?;
        if values.is_shallow_clone() {
            let result = values.try_promote_shared_claim(&permit);
            receipt.checkpoint(runtime).map_err(allocation)?;
            result.map_err(|source| match source {
                runmat_value::ExactArrayClaimPromotionError::Allocation(source) => {
                    allocation(source.into())
                }
                source => crate::runtime_error::semantic_error_with_source(
                    "RunMat:Call:Arguments",
                    "argument collection is not a canonical shared owner",
                    source,
                ),
            })?;
            values
        } else {
            let values = values
                .try_into_owned()
                .expect("unpublished collection is unique")
                .try_into_unbounded_vec()
                .expect("unpublished collection is ordinary");
            let result = OutputListStorage::try_adopt_exact(values, &permit);
            receipt.checkpoint(runtime).map_err(allocation)?;
            result.map_err(|source| allocation((*source.source()).into()))?
        }
    } else {
        ordinary(values)
    };
    if let Some(owner) = owner {
        owner.check(runtime).map_err(allocation)?;
    }
    Ok(result)
}
