//! Runtime adapter for admitted compound resource arithmetic. Execution owns
//! validation/admission; domains supply exact physical demand and currentness.
use super::RuntimeContext;
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, ProviderCapabilitySnapshot};
use runmat_execution::{
    CandidateOutputResidency, CandidateResourceDemand, ProviderInputStageDemand,
    ProviderInputStageDemands, ResourceAccount,
};

pub(super) fn preflight(
    runtime: Option<&RuntimeContext>,
    owner: &dyn AccelProvider,
    demand: CandidateResourceDemand,
    residency: CandidateOutputResidency,
    capabilities: &ProviderCapabilitySnapshot,
    validate_current: impl FnOnce() -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let parallel = runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let snapshot = owner.placement_resources();
    validate_current()?;
    runmat_execution::single_candidate_resources_admitted(
        runmat_execution::CandidateResourceFacts {
            location: runmat_execution::CandidateExecutionLocation::Provider {
                device_id: owner.device_id(),
            },
            output_residency: residency,
            presentation: &runmat_execution::CandidateOutputPresentation::Direct,
            kind: runmat_execution::ExecutionCandidateKind::ProviderOperation,
            demand,
            transactional_results: capabilities.concurrency.transactional_results,
        },
        &runmat_execution::PlacementResourceSnapshot {
            cpu_millicores_available: parallel.cpu_millicores_available,
            memory_available_bytes: parallel.memory_available_bytes,
            cancellation_requested: runtime.is_some_and(RuntimeContext::is_cancelled),
            providers: vec![snapshot],
            epoch: parallel.epoch,
        },
        false,
    )
    .map_err(super::super::provider_operation::resource_error)
}

pub(super) fn with_observation(
    resources: CandidateResourceDemand,
    observation: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
) -> BuiltinResult<CandidateResourceDemand> {
    let Some(observation) = observation else {
        return Ok(resources);
    };
    let bytes = observation
        .requested_peak_buffer_bytes
        .checked_add(
            observation
                .wgpu_compute_query
                .map_or(0, |query| query.requested_peak_buffer_bytes),
        )
        .ok_or_else(overflow)?;
    let add = |bound| {
        ResourceAccount::default()
            .add(bound)
            .and_then(|account| account.add(bytes.into()))
            .map(ResourceAccount::bound)
            .ok_or_else(overflow)
    };
    let mut resources = resources;
    resources.scratch_bytes = add(resources.scratch_bytes)?;
    if let Some(peak) = &mut resources.peak_live {
        peak.peak_bytes = peak.peak_bytes.checked_add(bytes).ok_or_else(overflow)?;
    }
    resources.provider_input_stages = resources
        .provider_input_stages
        .map(|stages| {
            stages
                .iter()
                .map(|stage| {
                    Ok(ProviderInputStageDemand {
                        incremental_peak_bytes: add(stage.incremental_peak_bytes)?,
                        ..stage
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()
                .and_then(|stages| {
                    ProviderInputStageDemands::sequential(stages).ok_or_else(overflow)
                })
        })
        .transpose()?;
    Ok(resources)
}

fn overflow() -> crate::RuntimeError {
    crate::build_runtime_error("compound observation resource extent overflows")
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
