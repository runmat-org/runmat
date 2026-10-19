//! Request-local stage metadata bound from the sealed source cardinalities.
//! Numeric buffers and provider-internal planning remain at their own owners.

use std::mem::{align_of, size_of};

use crate::fusion::semantics::preparation_resources::{
    ImageStageCardinality, PreparationBoundError,
};
use runmat_accelerate_api::{
    power_service::PowerTypedScalarProjection, GpuTensorHandle, ProviderImageOutputCall,
    ProviderOrderedRealMeanStep, ProviderRepresentation,
};
use runmat_runtime::context::host_allocation::HostAllocationPlan;

use super::{ordered_mean, selected_real_output, sigma_operations};

#[derive(Default)]
struct RequestedBytes(u64);

impl RequestedBytes {
    fn add_u64(&mut self, bytes: u64) -> Result<(), PreparationBoundError> {
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or(PreparationBoundError::Overflow)?;
        Ok(())
    }

    fn add(&mut self, bytes: usize) -> Result<(), PreparationBoundError> {
        let bytes = u64::try_from(bytes).map_err(|_| PreparationBoundError::Overflow)?;
        self.add_u64(bytes)
    }

    fn vector<T>(&mut self, count: usize) -> Result<(), PreparationBoundError> {
        self.add(
            count
                .checked_mul(size_of::<T>())
                .ok_or(PreparationBoundError::Overflow)?,
        )
    }

    fn boxed<T>(&mut self) -> Result<(), PreparationBoundError> {
        self.add(
            size_of::<T>()
                .checked_add(align_of::<T>())
                .ok_or(PreparationBoundError::Overflow)?,
        )
    }
}

/// All local stage proposal, placement-input and execution bookkeeping can
/// coexist with the sealed source/session projection. Counts are deliberately
/// summed across sequential hooks: no retained VM prefix is assumed dead.
pub(super) struct StageHostMemoryBound {
    pub scratch_bytes: u64,
    pub retained_view_bytes: u64,
}

pub(super) fn cold_cache_retained_bytes(
    rank: usize,
    metadata_payload: u64,
) -> Result<u64, PreparationBoundError> {
    let view =
        GpuTensorHandle::view_host_upper_bytes(rank).ok_or(PreparationBoundError::Overflow)?;
    let one = HostAllocationPlan::retained_view_bytes(view)
        .map_err(|_| PreparationBoundError::Overflow)?;
    let metadata = HostAllocationPlan::retained_metadata_bytes(metadata_payload)
        .map_err(|_| PreparationBoundError::Overflow)?;
    one.checked_mul(2)
        .and_then(|bytes| bytes.checked_add(HostAllocationPlan::retained_view_issuer_bytes().ok()?))
        .and_then(|bytes| bytes.checked_add(metadata))
        .ok_or(PreparationBoundError::Overflow)
}

pub(super) fn requested_bytes(
    cardinality: ImageStageCardinality,
) -> Result<StageHostMemoryBound, PreparationBoundError> {
    let ImageStageCardinality {
        source_steps,
        power_steps,
        power_identity_bytes,
        moment_cast_operation_bytes,
        sigma_fact_payload_bytes,
        moment_casts_upper,
        rank_upper,
    } = cardinality;
    let mut bytes = RequestedBytes::default();
    // The selected prepared state stays live while the final route is boxed;
    // both allocations are admitted before either can be constructed.
    bytes.boxed::<super::prepared::PreparedImageStages>()?;
    bytes.boxed::<super::vm::ImageVmStages>()?;
    let publication_views = source_steps
        .checked_add(2)
        .ok_or(PreparationBoundError::Overflow)?;
    let per_view = GpuTensorHandle::view_host_upper_bytes(rank_upper)
        .ok_or(PreparationBoundError::Overflow)?;
    let retained_per_view = HostAllocationPlan::retained_view_bytes(per_view)
        .map_err(|_| PreparationBoundError::Overflow)?;
    let retained_view_bytes = retained_per_view
        .checked_mul(u64::try_from(publication_views).map_err(|_| PreparationBoundError::Overflow)?)
        .and_then(|bytes| bytes.checked_add(HostAllocationPlan::retained_view_issuer_bytes().ok()?))
        .ok_or(PreparationBoundError::Overflow)?;
    bytes.add_u64(
        per_view
            .checked_mul(
                u64::try_from(moment_casts_upper).map_err(|_| PreparationBoundError::Overflow)?,
            )
            .ok_or(PreparationBoundError::Overflow)?,
    )?;
    bytes.vector::<usize>(rank_upper)?;
    // Two ordered reductions per moment, one sigma conversion route. Each
    // ordered reduction can plan two axes and retain two shape-bearing steps.
    const ORDERED_MEANS: usize = 4;
    const MOMENT_ROUTES: usize = 5;
    bytes.vector::<ordered_mean::OrderedMeanCallPlan>(ORDERED_MEANS)?;
    bytes.vector::<ordered_mean::SelectedMomentCastPlan>(moment_casts_upper)?;
    bytes.vector::<ProviderOrderedRealMeanStep>(MOMENT_ROUTES * 2)?;
    // Local axes and the provider-returned recipe control can coexist.
    bytes.vector::<usize>(MOMENT_ROUTES * 2 * 2)?;

    // Each cast feasibility call constructs one input and one output record
    // while the selected cast retains source and output representations.
    bytes.vector::<ProviderRepresentation>(
        moment_casts_upper
            .checked_mul(2)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;
    // The common cast owner constructs one feasibility operation String per
    // cast and one more while revalidating the live source against a selected
    // plan. These are canonical Single/Double names, not guessed text bytes.
    bytes.add(
        moment_casts_upper
            .checked_add(1)
            .and_then(|count| count.checked_mul(moment_cast_operation_bytes))
            .ok_or(PreparationBoundError::Overflow)?,
    )?;
    // Three selected binary calls are fixed; real output hooks are at most
    // source steps. Production diagnostic histories are cfg(test) only and
    // are not charged to a normal invocation.
    bytes.vector::<selected_real_output::PlannedCall>(source_steps)?;
    bytes.vector::<sigma_operations::SelectedSigmaReceipt>(source_steps)?;
    bytes.add_u64(sigma_fact_payload_bytes)?;

    // Two source-selected POWER calls retain symbolic plans. For each stage,
    // original, prospective replan and TypedDomain projection may coexist.
    // A symbolic plan owns at most four shape-bearing representations
    // (resident, two query inputs, one query output) and two usize shapes.
    // The typed pair adds two input plus two evaluation-output
    // representations and one probe usize shape. Thus at most 16
    // representations and seven usize shapes coexist per stage; the quoted
    // 18 and nine also cover transient feasibility/live-source projections.
    // Operation strings are the exact canonical POWER identity, not an
    // arbitrary fixed allowance.
    let power_plans = power_steps
        .checked_mul(3)
        .ok_or(PreparationBoundError::Overflow)?;
    // Only the two selected stages retain a boxed typed projection. Live
    // revalidation rebuilds symbolic POWER plans, not another typed box.
    let typed_projection_box_bytes = size_of::<PowerTypedScalarProjection>()
        .checked_add(align_of::<PowerTypedScalarProjection>())
        .and_then(|bytes| bytes.checked_mul(power_steps))
        .ok_or(PreparationBoundError::Overflow)?;
    bytes.add(typed_projection_box_bytes)?;
    bytes.vector::<ProviderRepresentation>(
        power_plans
            .checked_mul(6)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;
    bytes.vector::<usize>(
        power_plans
            .checked_mul(3)
            .and_then(|count| count.checked_mul(rank_upper))
            .ok_or(PreparationBoundError::Overflow)?,
    )?; // host/output shapes and the optional live view
    bytes.add_u64(
        power_identity_bytes
            .checked_mul(3)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;

    // OutputProgress retains one value slot and one sorted position per
    // source output. Scratch and published protections need no more than all
    // source steps plus the three fixed moments at one hook boundary.
    bytes.vector::<Option<GpuTensorHandle>>(source_steps)?;
    bytes.vector::<(u64, usize)>(source_steps)?;
    bytes.vector::<GpuTensorHandle>(6)?;
    bytes.vector::<&GpuTensorHandle>(
        source_steps
            .checked_add(3)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;

    // The pre-call physical candidate owns output representations, two
    // one-record source/prepared vectors, and one candidate record. Its
    // workload clone is already included in source projected-program bytes.
    bytes.vector::<ProviderRepresentation>(
        source_steps
            .checked_add(4)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;
    bytes.vector::<crate::placement::LocalPhysicalCandidate<'static>>(1)?;
    bytes.add("cpu.shared.fusion.image_normalize".len())?;
    // The one resident source representation is built from a temporary
    // `shape().to_vec()` before conversion into its retained u64 shape.
    bytes.vector::<usize>(rank_upper)?;

    // Heap payloads of all retained ProviderRepresentation shapes and
    // transient cast/mean proposals. The site counts are: 25 ordered-mean
    // source/step shapes plus four temporary expected/output projections per
    // moment route; six per cast (proposal, query and retained source/
    // output); six per fixed binary; four per selected real/output call;
    // four per selected POWER; and one per pre-call output/source record.
    let shape_copies = (25 + MOMENT_ROUTES * 4)
        .checked_add(
            moment_casts_upper
                .checked_mul(6)
                .ok_or(PreparationBoundError::Overflow)?,
        )
        .and_then(|count| count.checked_add(3 * 6 + 2 * 4))
        .and_then(|count| count.checked_add(source_steps.checked_mul(5)?))
        .and_then(|count| count.checked_add(power_plans.checked_mul(6)?))
        .and_then(|count| count.checked_add(4))
        .ok_or(PreparationBoundError::Overflow)?;
    bytes.vector::<u64>(
        shape_copies
            .checked_mul(rank_upper)
            .ok_or(PreparationBoundError::Overflow)?,
    )?;

    // The typed output-program call structs are held by source projection;
    // this one additional live temporary may be built for stage validation.
    bytes.vector::<ProviderImageOutputCall>(source_steps)?;
    Ok(StageHostMemoryBound {
        scratch_bytes: bytes.0,
        retained_view_bytes,
    })
}

/// Incremental request-local host-source metadata before a selected image
/// upload. The common stage quote above owns the selected recipes and generic
/// physical candidate; this counts only host-source COW shape, upload-query,
/// route and subsequent host-specific comparison projections.
pub(super) fn host_source_requested_bytes(rank: usize) -> Result<u64, PreparationBoundError> {
    let mut bytes = RequestedBytes::default();
    // The host-source plan is boxed before placement, then consumed by the
    // one-shot upload. Count its allocation even though it normally retires
    // before the selected prepared-state box is installed.
    bytes.boxed::<super::host_source::HostSingleImageSource>()?;
    bytes.add_u64(
        GpuTensorHandle::view_host_upper_bytes(rank).ok_or(PreparationBoundError::Overflow)?,
    )?;
    // Source binding retains one host Tensor; selected planning constructs a
    // transient Value::Tensor and retains its own source. All three clone a
    // shape but share the authoritative COW numeric payload.
    bytes.vector::<usize>(rank.checked_mul(3).ok_or(PreparationBoundError::Overflow)?)?;
    // `ordered_mean::representation`, upload query input/output, the retained
    // upload route source/output, and three host-specific placement/preflight
    // representation clones each own a shape vector.
    bytes.vector::<u64>(rank.checked_mul(8).ok_or(PreparationBoundError::Overflow)?)?;
    bytes.vector::<ProviderRepresentation>(2)?; // upload query inputs/outputs
    bytes.add("transfer.upload".len())?; // ProviderOperationIdentity
    Ok(bytes.0)
}
