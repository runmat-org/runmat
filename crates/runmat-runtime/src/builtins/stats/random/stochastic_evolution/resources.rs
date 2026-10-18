//! Source-loop resource geometry. These are pure quotes, not allocation grants.
//! The selected output constructors retain their own exact receipts; the
//! coordinator must not admit these same outputs a second time.

use super::host;
use super::provider::{InitialProviderSource, StochasticNoiseUploadPlan};
use super::source_eligibility::StochasticTypedPattern;
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::context::host_allocation::{provider_registration_host_quote, HostAllocationPlan};
use crate::context::host_presentation::HostPresentationReceipt;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::observation::{
    ProviderOperationReceipt, ProviderResourceScope, ProviderResourceTicket,
};
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionSelection, PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use runmat_accelerate_api::{AccelProvider, ProviderQueryMetadataFootprint};
use runmat_execution::{
    CandidatePeakLiveDemand, CandidateResourceDemand, ProviderInputStageDemand,
    ProviderInputStageDemands, ResourceAccount, ResourceBoundUnavailable, ResourceByteBound,
};

fn quote_error() -> RuntimeError {
    build_runtime_error("stochastic source resource geometry exceeds its representation")
        .with_identifier("RunMat:StochasticSource:ResourceGeometry")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

/// Pre-admit the exact caller-owned metadata that can coexist at the end of
/// provider planning: two shape Vecs in the work request, two in the selected
/// plan, and two in the feasibility query, plus the latter's representation
/// Vec slots and operation String. Provider-internal planning workspace is a
/// separate provider-owned bound added by the caller before admission.
pub(super) fn planning_metadata_plan(rank: usize) -> BuiltinResult<HostAllocationPlan> {
    let footprint = ProviderQueryMetadataFootprint {
        operation: PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
        input_ranks: &[rank],
        output_ranks: &[rank],
        preparation_shape_ranks: &[rank, rank, rank, rank],
        preparation_representation_slots: 0,
    };
    let bytes = footprint.requested_bytes().ok_or_else(quote_error)?;
    HostAllocationPlan::new(0, bytes).map_err(|_| quote_error())
}

/// Final Z/S values are retained; at a later iteration either the old and new
/// Z overlap during generation, or old and new S overlap during arithmetic.
/// Those phases are sequential, not two simultaneous copies of both outputs.
/// The initial prepared state is already live in the canonical ledger.
#[derive(Clone, Copy, Debug)]
pub(super) struct StochasticHostResources {
    pub(super) result_bytes: ResourceByteBound,
    pub(super) scratch_bytes: ResourceByteBound,
    pub(super) noise_output_bytes: u64,
}

pub(super) fn host_resources(
    pattern: &StochasticTypedPattern,
    mode: ExactHostRandomMode,
    publication_host_workspace: ResourceByteBound,
) -> BuiltinResult<StochasticHostResources> {
    let noise_bytes = host::output_bytes(pattern.noise_class(), pattern.shape(), mode)?;
    let state_bytes = host::output_bytes(pattern.result_class(), pattern.shape(), mode)?;
    let fresh_pair = noise_bytes
        .checked_add(state_bytes)
        .ok_or_else(quote_error)?;
    let replacement_peak = if pattern.steps() > 1 {
        noise_bytes.max(state_bytes)
    } else {
        0
    };
    let scratch_bytes = ResourceAccount::default()
        .checked_add(ResourceByteBound::Known(replacement_peak))
        .and_then(|account| account.checked_add(publication_host_workspace))
        .map(ResourceAccount::bound)
        .ok_or_else(quote_error)?;
    Ok(StochasticHostResources {
        result_bytes: ResourceByteBound::Known(fresh_pair),
        // Numeric constructors fill final buffers directly; replacement and
        // canonical VM publication are the only extra phase obligations here.
        // MT state mutation and its claim belong to RandomTurn.
        scratch_bytes,
        noise_output_bytes: noise_bytes,
    })
}

/// One serial tile's provider-owned phases. The upload output is a physical
/// buffer, not part of its scratch scope; compute output registration is also
/// separately owned from the uniform and completion-fence tickets.
pub(super) struct StochasticProviderResources {
    pub(super) demand: CandidateResourceDemand,
    pub(super) metadata_registration_count: usize,
    pub(super) phase_limit_bytes: u64,
    pub(super) upload_phase_limit_bytes_initial: Option<ResourceByteBound>,
    pub(super) upload_phase_limit_bytes_full: ResourceByteBound,
    pub(super) upload_phase_limit_bytes_remainder: Option<ResourceByteBound>,
    pub(super) host_plan: HostAllocationPlan,
    /// Raw payload of the one retained provider-observation operation name.
    /// Whole-work selected/deferred identity strings have a separate
    /// pre-selection owner at the placement boundary.
    pub(super) publication_metadata_payload_bytes: u64,
}

fn known_sum(values: impl IntoIterator<Item = u64>) -> BuiltinResult<u64> {
    values
        .into_iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(value))
        .ok_or_else(quote_error)
}

fn upload_peak(
    phase: &super::provider::StochasticNoiseUploadPhase,
    live_state_bytes: u64,
) -> BuiltinResult<ResourceByteBound> {
    ResourceAccount::default()
        .checked_add(ResourceByteBound::Known(known_sum([
            live_state_bytes,
            phase.physical.output_requested_bytes,
        ])?))
        .and_then(|account| account.checked_add(phase.physical.provider_scratch_bytes))
        .map(ResourceAccount::bound)
        .ok_or_else(quote_error)
}

fn upload_stage(
    provider: &dyn AccelProvider,
    phase: &super::provider::StochasticNoiseUploadPhase,
    live_state_bytes: u64,
    already_live_bytes: u64,
) -> BuiltinResult<ProviderInputStageDemand> {
    let absolute = upload_peak(phase, live_state_bytes)?;
    let incremental = match absolute {
        ResourceByteBound::Known(bytes) => ResourceByteBound::Known(
            bytes
                .checked_sub(already_live_bytes)
                .ok_or_else(quote_error)?,
        ),
        ResourceByteBound::Unavailable(reason) => ResourceByteBound::Unavailable(reason),
    };
    Ok(ProviderInputStageDemand {
        device_id: provider.device_id(),
        incremental_peak_bytes: incremental,
        queue_slots: phase.physical.completion_queue_slots,
    })
}

/// Quote one selected source loop without recreating provider tile geometry or
/// upload evidence. The caller supplies VM publication metadata evidence: an
/// unavailable bound stays unavailable rather than being promoted to zero.
/// Initial prepared state and planning/upload-query receipts are already live
/// in the shared ledger and are not charged again to the host plan.
pub(super) fn provider_resources(
    provider: &dyn AccelProvider,
    selected: &ProviderStochasticEvolutionSelection,
    initial_source: &InitialProviderSource,
    upload: &StochasticNoiseUploadPlan,
    host: &StochasticHostResources,
    publication_host_workspace: ResourceByteBound,
    cpu_millicores: u32,
) -> BuiltinResult<Option<StochasticProviderResources>> {
    if !selected.proposal.validate_for(&selected.plan) {
        return Err(quote_error());
    }
    let proposal = selected.proposal;
    let Some(registration_pool_payload) = provider.reusable_output_registration_host_bytes() else {
        return Ok(None);
    };
    let registration_pool_claim =
        HostAllocationPlan::retained_metadata_bytes(registration_pool_payload)
            .map_err(|_| quote_error())?;
    let rank = selected.plan.output.shape.len();
    let simultaneous_outputs = if selected.plan.steps > 1 { 2 } else { 1 };
    // One noise tile and the source-loop's old/new S records can coexist.
    // A host source needs one additional uploaded initial-state record;
    // a resident source's original record is already live in the ledger.
    let metadata_registration_count = simultaneous_outputs
        + 1
        + usize::from(matches!(
            initial_source,
            InitialProviderSource::HostUpload(_)
        ));
    let Some(output_registration) =
        provider_registration_host_quote(provider, rank, simultaneous_outputs, 0)
            .map_err(|_| quote_error())?
    else {
        return Ok(None);
    };
    let Some(noise_registration) = upload.full.registration else {
        return Ok(None);
    };
    let remainder_registration = match upload.remainder.as_ref() {
        Some(phase) => Some(phase.registration.ok_or_else(quote_error)?),
        None => None,
    };
    let initial_registration = match initial_source {
        InitialProviderSource::Resident(_) => None,
        InitialProviderSource::HostUpload(plan) => {
            Some(plan.get().phase.registration.ok_or_else(quote_error)?)
        }
    };
    // The registration quote includes this exact owned view. Adding the
    // proposal's view field again would double-charge one shape allocation.
    let Some(view) = provider.registered_output_host_metadata(rank) else {
        return Ok(None);
    };
    if proposal.output_view_host_bytes != view.view_payload_bytes {
        return Err(quote_error());
    }
    let phase_limit_bytes = known_sum([
        proposal.uniform_buffer_bytes_per_tile,
        proposal.completion_fence_bytes_per_tile,
        proposal.observation_fence_bytes_per_tile,
    ])?;
    let transient_host = known_sum([
        proposal.authentication_host_bytes_per_tile,
        output_registration
            .preparation_bytes
            .max(noise_registration.preparation_bytes)
            .max(remainder_registration.map_or(0, |r| r.preparation_bytes))
            .max(initial_registration.map_or(0, |r| r.preparation_bytes)),
    ])?;
    let retained_control =
        HostAllocationPlan::retained_metadata_bytes(proposal.completion_control_host_bytes)
            .map_err(|_| quote_error())?;
    let ticket = ProviderResourceTicket::host_metadata_upper_bytes().ok_or_else(quote_error)?;
    // Transfer retirement is acknowledged before compute/next upload. Its
    // one fence ticket does not overlap the three compute tickets (uniform,
    // semantic fence, observed-submission fence).
    let transfer_callback = known_sum([proposal.source_transfer_retirement_host_bytes, ticket])?;
    let compute_callback = known_sum([
        proposal.submission_retention_host_bytes_per_tile,
        ticket.checked_mul(3).ok_or_else(quote_error)?,
    ])?;
    let callback_control = transfer_callback
        .max(compute_callback)
        .checked_add(proposal.observation_host_control_bytes_per_tile)
        .ok_or_else(quote_error)?;
    // Each rank-specific quote includes its provider table-growth upper. The
    // shared table may reuse backing, so their sum is a conservative physical
    // upper, not multiple independent table grants or an exact RSS reading.
    let publication_metadata_payload_bytes = (PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION.len()
        as u64)
        .checked_add(
            HostPresentationReceipt::managed_retention_host_upper_bytes()
                .ok_or_else(quote_error)?,
        )
        .and_then(|bytes| {
            bytes.checked_add(ProviderOperationReceipt::with_budget_host_metadata_upper_bytes()?)
        })
        .and_then(|bytes| {
            bytes
                .checked_add(ProviderResourceScope::first_single_phase_host_metadata_upper_bytes()?)
        })
        .and_then(|bytes| bytes.checked_add(callback_control))
        .ok_or_else(quote_error)?;
    let publication_metadata_claim_bytes =
        HostAllocationPlan::retained_metadata_bytes(publication_metadata_payload_bytes)
            .map_err(|_| quote_error())?;
    let retained_host = known_sum([
        retained_control,
        registration_pool_claim,
        publication_metadata_claim_bytes,
        output_registration.retained_bytes,
        noise_registration.retained_bytes,
        initial_registration.map_or(0, |r| r.retained_bytes),
    ])?;
    let host_plan =
        HostAllocationPlan::new(transient_host, retained_host).map_err(|_| quote_error())?;
    let provider_host_workspace = ResourceAccount::default()
        .checked_add(ResourceByteBound::Known(known_sum([
            // ProviderAndHost presentation independently adds the final Z.
            // Only its old/new replacement overlap belongs in workspace.
            if selected.plan.steps > 1 {
                host.noise_output_bytes
            } else {
                0
            },
            host_plan.scratch_peak_bytes(),
            host_plan.retained_output_bytes(),
        ])?))
        .and_then(|account| account.checked_add(publication_host_workspace))
        .map(ResourceAccount::bound)
        .ok_or_else(quote_error)?;

    let (initial, already_live, initial_upload) = match initial_source {
        InitialProviderSource::Resident(handle) => {
            let actual = handle.physical().requested_bytes();
            if actual
                < selected
                    .plan
                    .state
                    .checked_byte_len()
                    .ok_or_else(quote_error)?
            {
                return Err(quote_error());
            }
            (actual, actual, None)
        }
        InitialProviderSource::HostUpload(plan) => {
            let plan = plan.get();
            let actual = plan.phase.physical.output_requested_bytes;
            if actual
                < selected
                    .plan
                    .state
                    .checked_byte_len()
                    .ok_or_else(quote_error)?
            {
                return Err(quote_error());
            }
            (actual, 0, Some(&plan.phase))
        }
    };
    let prior_created = if selected.plan.steps > 1 {
        proposal
            .first_output_buffer_bytes
            .max(proposal.continuing_output_buffer_bytes)
    } else {
        0
    };
    let max_output = if selected.plan.steps > 1 {
        proposal
            .first_output_buffer_bytes
            .max(proposal.continuing_output_buffer_bytes)
    } else {
        proposal.first_output_buffer_bytes
    };
    let live_state = known_sum([initial, prior_created])?;
    let mut peak = ResourceAccount::default()
        .checked_add(ResourceByteBound::Known(known_sum([
            live_state,
            upload.full.physical.output_requested_bytes,
            max_output,
            phase_limit_bytes,
        ])?))
        .ok_or_else(quote_error)?;
    for phase in [&upload.full].into_iter().chain(upload.remainder.iter()) {
        peak = peak.peak(
            ResourceAccount::default()
                .checked_add(upload_peak(phase, live_state)?)
                .ok_or_else(quote_error)?,
        );
    }
    if let Some(phase) = initial_upload {
        peak = peak.peak(
            ResourceAccount::default()
                .checked_add(upload_peak(phase, 0)?)
                .ok_or_else(quote_error)?,
        );
    }
    let peak_live = match peak.bound() {
        ResourceByteBound::Known(bytes) => Some(CandidatePeakLiveDemand {
            peak_bytes: bytes,
            already_live_bytes: already_live,
        }),
        ResourceByteBound::Unavailable(_) => None,
    };
    let scratch_bytes = match peak_live {
        Some(peak) => ResourceByteBound::Known(peak.incremental_bytes().ok_or_else(quote_error)?),
        None => ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
    };
    let queue_slots = upload
        .full
        .physical
        .completion_queue_slots
        .max(
            upload
                .remainder
                .as_ref()
                .map_or(0, |p| p.physical.completion_queue_slots),
        )
        .max(initial_upload.map_or(0, |p| p.physical.completion_queue_slots))
        .max(1);
    let initial_stage = initial_upload
        .map(|phase| upload_stage(provider, phase, 0, already_live))
        .transpose()?;
    let full_stage = upload_stage(provider, &upload.full, live_state, already_live)?;
    let remainder_stage = upload
        .remainder
        .as_ref()
        .map(|phase| upload_stage(provider, phase, live_state, already_live))
        .transpose()?;
    let provider_input_stages = ProviderInputStageDemands::sequential(
        initial_stage
            .into_iter()
            .chain(std::iter::once(full_stage))
            .chain(remainder_stage),
    )
    .ok_or_else(quote_error)?;
    Ok(Some(StochasticProviderResources {
        demand: CandidateResourceDemand {
            cpu_millicores,
            retained_bytes: ResourceByteBound::Known(proposal.final_output_buffer_bytes),
            scratch_bytes,
            provider_host_workspace: Some(provider_host_workspace),
            provider_input_stages: Some(provider_input_stages),
            queue_slots,
            peak_live,
        },
        metadata_registration_count,
        phase_limit_bytes,
        upload_phase_limit_bytes_initial: initial_upload
            .map(|phase| phase.physical.provider_scratch_bytes),
        upload_phase_limit_bytes_full: upload.full.physical.provider_scratch_bytes,
        upload_phase_limit_bytes_remainder: upload
            .remainder
            .as_ref()
            .map(|phase| phase.physical.provider_scratch_bytes),
        host_plan,
        publication_metadata_payload_bytes,
    }))
}
