//! Metadata-only reshape retains the original slot/provenance/class owners.
use super::*;
use runmat_value::{ExactArrayClaimPromotionError, SharedShape};

pub(crate) fn reshape(
    source: ObjectArray,
    shape: &[usize],
    runtime: Option<&RuntimeContext>,
) -> Result<ObjectArray, RuntimeError> {
    if shape.len() < 2
        || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(source.len())
    {
        return Err(build_error(
            runtime,
            ObjectArrayBuildError::ShapeCountMismatch,
        ));
    }
    let owner =
        HostAllocationReceipt::capture_owner(runtime).map_err(|e| allocation(runtime, e))?;
    let _roots =
        super::super::structure::replacement::lease(&Value::ObjectArray(source.clone()), runtime)?;
    let data = source
        .data_owner()
        .unclaimed_shared_bytes()
        .map_err(|e| promotion(runtime, e))?;
    let provenance = source
        .provenance_owner()
        .unclaimed_shared_bytes()
        .map_err(|e| promotion(runtime, e))?;
    let class = if source.class_name_owner().has_retained_claim() {
        None
    } else {
        Some(
            source
                .class_name_owner()
                .requested_bytes()
                .map_err(|e| allocation(runtime, e.into()))?,
        )
    };
    let mut bytes = SharedShape::requested_bytes_for_rank(shape.len())
        .map_err(|e| allocation(runtime, e.into()))?;
    let mut claims = 1usize;
    for extra in [data, provenance, class].into_iter().flatten() {
        bytes = bytes
            .checked_add(extra)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        claims += 1;
    }
    let output = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|e| allocation(runtime, e))?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, output).map_err(|e| allocation(runtime, e))?,
    )
    .map_err(|e| allocation(runtime, e))?;
    let shape = if let Some(receipt) = &receipt {
        let context = runtime.expect("managed selected context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        source
            .data_owner()
            .try_promote_shared_claim(&permit)
            .map_err(|e| promotion(runtime, e))?;
        source
            .provenance_owner()
            .try_promote_shared_claim(&permit)
            .map_err(|e| promotion(runtime, e))?;
        source
            .class_name_owner()
            .clone()
            .try_admit_exact(&permit)
            .map_err(|e| allocation(runtime, e.source().into()))?;
        let shape = SharedShape::new_admitted_from_slice(shape, &permit)
            .map_err(|e| allocation(runtime, e.into()))?;
        receipt
            .checkpoint(context)
            .map_err(|e| allocation(runtime, e))?;
        shape
    } else {
        SharedShape::new_unmanaged(shape.to_vec())
    };
    let result = source
        .try_with_shape(shape)
        .map_err(|e| crate::runtime_error::semantic_error("ObjectArrayConstruction", e))?;
    if let Some(owner) = owner {
        owner
            .check(runtime.expect("selected context"))
            .map_err(|e| allocation(runtime, e))?;
    }
    Ok(result)
}
fn promotion(
    runtime: Option<&RuntimeContext>,
    source: ExactArrayClaimPromotionError,
) -> RuntimeError {
    match source {
        ExactArrayClaimPromotionError::Allocation(source) => allocation(runtime, source.into()),
        source => crate::runtime_error::semantic_error_with_source(
            "ObjectArrayConstruction",
            "object array storage is not a canonical shared owner",
            source,
        ),
    }
}
