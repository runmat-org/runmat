//! One complete, exact provider operation admitted by the shared whole-work
//! service. Domain owners supply the typed query and physical host workspace;
//! this boundary owns policy selection, the provider fence, and feedback.

use super::{
    host_presentation::{combine_host_scratch, HostPresentationReceipt},
    RuntimeContext,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport, ProviderOperationRecorder,
};
use runmat_accelerate_api::{
    provider_for_handle, AccelProvider, GpuTensorHandle, ProviderFeasibility,
    ProviderFeasibilityQuery,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePeakLiveDemand,
    CandidatePreparationState, CandidateResourceDemand, Digest, ExecutionCandidateKind,
    PlacementResourceSnapshot, PlacementSignature, ProviderInputStageDemand,
    ProviderInputStageDemands, ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkPlacementRequest,
};

/// An original resident operand whose native readback occurs during this
/// provider-first route. `prior_created_bytes` is operation-created provider
/// storage still live during this stage, excluding authenticated borrowed
/// allocations already in that owner's resource snapshot.
pub(crate) struct ExactProviderReadback<'a> {
    pub(crate) source: &'a GpuTensorHandle,
    pub(crate) prior_created_bytes: ResourceByteBound,
    pub(crate) phase: ExactProviderReadbackPhase,
}

#[derive(Clone, Copy, Eq, PartialEq, serde::Serialize)]
pub(crate) enum ExactProviderReadbackPhase {
    Preparation,
    Conditional,
}

/// Physical continuation admitted only for an Unsupported provider hook that
/// computes on host and restores to this same original provider owner.
#[derive(Clone, Copy)]
pub(crate) struct ExactConditionalProviderUpload {
    pub(crate) provider_output_requested_bytes: u64,
    pub(crate) transfer_bytes: Option<u64>,
    pub(crate) host_result_bytes: ResourceByteBound,
    pub(crate) host_scratch_bytes: ResourceByteBound,
    pub(crate) provider_scratch_bytes: ResourceByteBound,
    pub(crate) queue_slots: u32,
}

pub(crate) struct ExactProviderOperation<'a> {
    pub(crate) provider: &'static dyn AccelProvider,
    pub(crate) query: &'a ProviderFeasibilityQuery,
    pub(crate) semantic_operation: &'static str,
    pub(crate) variant: &'static str,
    /// Typed controls and original operand provenance. Never a textual
    /// substitute for the live handles that the domain owner authenticates.
    pub(crate) domain_facts: Digest,
    /// Peak operation-owned host workspace, including any work before the
    /// provider hook but excluding provider internal host workspace.
    pub(crate) host_scratch: ResourceByteBound,
    /// Provider-hook internal host workspace, certified by its typed physical
    /// route contract. This remains separate from operation-owned preparation.
    pub(crate) provider_host_scratch: ResourceByteBound,
    pub(crate) host_cpu_millicores: u32,
    /// Physical bytes of authenticated borrowed inputs already represented
    /// in the provider's live-allocation resource snapshot.
    pub(crate) already_live_bytes: u64,
    pub(crate) readbacks: &'a [ExactProviderReadback<'a>],
    /// Provider-owned internal stages not represented by original readbacks
    /// (for example a second readback of an adapted, newly uploaded kernel).
    /// The typed provider operation contract must prove these stage peaks.
    pub(crate) internal_input_stages: Option<ProviderInputStageDemands>,
    pub(crate) output_queue_slots: Option<u32>,
    /// The public form permits a host result only after this provider hook
    /// returns Unsupported. This is one conditional route, not an equivalent
    /// host alternative offered to placement policy.
    pub(crate) conditional_host_fallback: Option<ResourceByteBound>,
    /// Fallback-only host workspace; includes its final host result when that
    /// result is published on host, unlike provider upload's exclusive scratch.
    pub(crate) conditional_host_scratch: ResourceByteBound,
    pub(crate) conditional_provider_upload: Option<ExactConditionalProviderUpload>,
}

pub(crate) struct SelectedProviderOperation {
    provider: &'static dyn AccelProvider,
    receipt: HostPresentationReceipt,
    conditional_output_requested_bytes: Option<u64>,
}

impl SelectedProviderOperation {
    pub(crate) fn begin_conditional_provider_upload(&mut self) {
        self.receipt.begin_conditional_provider_upload();
    }

    pub(crate) fn recorder(&self) -> ProviderOperationRecorder {
        self.receipt.recorder()
    }

    pub(crate) fn provider(&self) -> &'static dyn AccelProvider {
        self.provider
    }

    pub(crate) fn admits_conditional_provider_upload(&self, output: &GpuTensorHandle) -> bool {
        self.conditional_output_requested_bytes
            .is_some_and(|bytes| output.physical().requested_bytes() == bytes)
            && output.device_id() == self.provider.device_id()
            && output.allocation_lease().validate_live().is_ok()
    }

    pub(crate) fn finish(
        self,
        published: Result<bool, runmat_accelerate_api::observation::ProviderObservationFailure>,
    ) {
        self.receipt.finish(published);
    }

    pub(crate) fn finish_conditional_host_fallback(self) {
        self.receipt.finish_conditional_host_fallback();
    }

    pub(crate) fn finish_conditional_provider_upload(self) {
        self.receipt.finish_conditional_provider_upload();
    }
}

pub(crate) fn select_exact_provider_operation(
    runtime: Option<&RuntimeContext>,
    request: ExactProviderOperation<'_>,
) -> BuiltinResult<Option<SelectedProviderOperation>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let provider = request.provider;
    let query = request.query;
    if request.conditional_host_fallback.is_some() && request.conditional_provider_upload.is_some()
    {
        return Err(build_runtime_error(
            "provider operation has conflicting conditional publications",
        )
        .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
        .build());
    }
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(query) else {
        return Ok(None);
    };
    if request
        .conditional_provider_upload
        .as_ref()
        .is_some_and(|upload| estimate.output_bytes != Some(upload.provider_output_requested_bytes))
    {
        return Err(build_runtime_error(
            "conditional provider upload has a different physical output bound",
        )
        .with_identifier("RunMat:Placement:InvalidConditionalPresentation")
        .build());
    }
    let ProviderObservationCapability::Supported(observation) =
        provider.operation_observation_capability(query)
    else {
        return Ok(None);
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let mut provider_resources = vec![provider.placement_resources()];
    let mut source_capabilities = Vec::with_capacity(request.readbacks.len());
    let mut readback_facts = Vec::with_capacity(request.readbacks.len());
    let mut input_stages = Vec::new();
    let mut preparation_host_workspace = ResourceByteBound::Known(0);
    let mut conditional_readback_host_workspace = ResourceByteBound::Known(0);
    for readback in request.readbacks {
        let source = readback.source;
        let Some(source_provider) = provider_for_handle(source) else {
            return Err(
                build_runtime_error("provider input readback has no exact owner")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
            );
        };
        source.allocation_lease().validate_live().map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                build_runtime_error("provider input readback allocation is unavailable")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        if source.device_id() != source_provider.device_id() {
            return Err(
                build_runtime_error("provider input readback owner device is invalid")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
            );
        }
        let source_resource = source_provider.placement_resources();
        let Some(download) = ProviderFeasibilityQuery::exact_numeric_download(source) else {
            if parallel.memory_available_bytes.is_some()
                || source_resource.capacity_bytes.is_some()
                || source_resource.scratch_available_bytes.is_some()
            {
                return Err(build_runtime_error(
                    "provider input has no exact native readback under a finite resource budget",
                )
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build());
            }
            return Ok(None);
        };
        if !source_provider
            .query_numeric_download_feasibility(source, &download)
            .is_supported()
        {
            if parallel.memory_available_bytes.is_some()
                || source_resource.capacity_bytes.is_some()
                || source_resource.scratch_available_bytes.is_some()
            {
                return Err(build_runtime_error(
                    "provider input readback has no proven route under a finite resource budget",
                )
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build());
            }
            return Ok(None);
        }
        let workspace = source_provider.download_numeric_host_workspace_bound(source);
        match readback.phase {
            ExactProviderReadbackPhase::Preparation => {
                preparation_host_workspace =
                    combine_host_scratch(preparation_host_workspace, workspace)?;
            }
            ExactProviderReadbackPhase::Conditional => {
                conditional_readback_host_workspace =
                    combine_host_scratch(conditional_readback_host_workspace, workspace)?;
            }
        }
        let stage_peak = sum_resource_bounds(
            readback.prior_created_bytes,
            readback.phase,
            source_provider.download_numeric_provider_scratch_bound(source),
        )?;
        input_stages.push(ProviderInputStageDemand {
            device_id: source_provider.device_id(),
            incremental_peak_bytes: stage_peak,
            queue_slots: source_provider.download_numeric_queue_slots(source),
        });
        source_capabilities.push((
            source_provider.device_id(),
            source_provider.capability_snapshot(),
        ));
        readback_facts.push((
            download,
            readback.prior_created_bytes,
            source.view().layout(),
            runmat_accelerate_api::handle_provenance(source),
        ));
        if !provider_resources
            .iter()
            .any(|resource| resource.device_id == source_resource.device_id)
        {
            provider_resources.push(source_resource);
        }
    }
    if let Some(internal) = request.internal_input_stages {
        if !internal.is_canonical() {
            return Err(
                build_runtime_error("provider input stage set is not canonical")
                    .with_identifier("RunMat:Placement:InvalidInputStages")
                    .build(),
            );
        }
        if internal
            .iter()
            .any(|stage| stage.device_id != provider.device_id())
        {
            return Err(
                build_runtime_error("provider internal stage has a foreign owner")
                    .with_identifier("RunMat:Placement:InvalidInputStages")
                    .build(),
            );
        }
        input_stages.extend(internal.iter());
    }
    let input_stages = if input_stages.is_empty() {
        None
    } else {
        Some(
            ProviderInputStageDemands::sequential(input_stages).ok_or_else(|| {
                build_runtime_error("provider input stages exceed the resource set")
                    .with_identifier("RunMat:Placement:TooManyProviders")
                    .build()
            })?,
        )
    };
    let prefix_host_workspace =
        combine_host_scratch(request.host_scratch, preparation_host_workspace)?;
    let direct_host_peak =
        combine_host_scratch(prefix_host_workspace, request.provider_host_scratch)?;
    let host_scratch = if request.conditional_host_fallback.is_some() {
        let fallback = combine_host_scratch(
            combine_host_scratch(prefix_host_workspace, conditional_readback_host_workspace)?,
            request.conditional_host_scratch,
        )?;
        peak_resource_bounds(direct_host_peak, fallback)
    } else {
        direct_host_peak
    };
    let conditional_provider_upload = if let Some(upload) = request.conditional_provider_upload {
        Some(ExactConditionalProviderUpload {
            host_scratch_bytes: combine_host_scratch(
                combine_host_scratch(prefix_host_workspace, conditional_readback_host_workspace)?,
                upload.host_scratch_bytes,
            )?,
            ..upload
        })
    } else {
        None
    };
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&(provider.capability_snapshot(), &source_capabilities))
            .expect("capability snapshot serializes"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let signature = PlacementSignature {
        region: None,
        operation: request.semantic_operation.into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.variant,
                request.semantic_operation,
                request.domain_facts,
                query,
                &readback_facts,
                input_stages,
                host_scratch,
                preparation_host_workspace,
                conditional_readback_host_workspace,
                request.output_queue_slots,
                request.conditional_host_fallback,
                request.conditional_host_scratch,
                conditional_provider_upload.as_ref().map(|upload| {
                    (
                        upload.transfer_bytes,
                        upload.provider_output_requested_bytes,
                        upload.host_result_bytes,
                        upload.host_scratch_bytes,
                        upload.provider_scratch_bytes,
                        upload.queue_slots,
                    )
                }),
                provider.device_id(),
            ))
            .expect("typed provider operation serializes"),
        ),
        revision,
    };
    // Completion fences coexist with the operation's output/temporary peak.
    // An overflow is unknown physical demand, never a wrapped small bound.
    let peak_live = estimate
        .transient_bytes
        .and_then(|bytes| bytes.checked_add(observation.requested_peak_buffer_bytes))
        .map(|peak_bytes| CandidatePeakLiveDemand {
            peak_bytes,
            already_live_bytes: request.already_live_bytes,
        });
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::ProviderOperation,
        execution_location: CandidateExecutionLocation::Provider {
            device_id: provider.device_id(),
        },
        preparation: CandidatePreparationState::Ready,
        output_residency: if let Some(host_bytes) = request.conditional_host_fallback {
            CandidateOutputResidency::ConditionalHostOrProvider {
                device_id: provider.device_id(),
                host_bytes,
            }
        } else {
            CandidateOutputResidency::Provider {
                device_id: provider.device_id(),
            }
        },
        presentation: conditional_provider_upload.as_ref().map_or(
            CandidateOutputPresentation::Direct,
            |upload| CandidateOutputPresentation::ConditionalProviderUpload {
                device_id: provider.device_id(),
                transfer_bytes: upload.transfer_bytes,
                host_result_bytes: Some(upload.host_result_bytes),
                host_scratch_bytes: upload.host_scratch_bytes,
                provider_scratch_bytes: upload.provider_scratch_bytes,
                queue_slots: upload.queue_slots,
            },
        ),
        route_facts: Digest::sha256(
            serde_json::to_vec(&(
                query,
                &estimate,
                request.conditional_host_fallback,
                conditional_provider_upload.as_ref().map(|upload| {
                    (
                        upload.transfer_bytes,
                        upload.provider_output_requested_bytes,
                        upload.host_result_bytes,
                        upload.host_scratch_bytes,
                        upload.provider_scratch_bytes,
                        upload.queue_slots,
                    )
                }),
                input_stages,
                request.output_queue_slots,
            ))
            .expect("provider route serializes"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: request.host_cpu_millicores,
            retained_bytes: estimate
                .output_bytes
                .map(ResourceByteBound::Known)
                .unwrap_or(ResourceByteBound::Unavailable(
                    ResourceBoundUnavailable::NoRetainedValueBound,
                )),
            scratch_bytes: ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            provider_host_workspace: Some(host_scratch),
            provider_input_stages: input_stages,
            queue_slots: request
                .output_queue_slots
                .unwrap_or_else(|| estimate.dispatches.unwrap_or(1)),
            peak_live,
        },
        transactional_results: provider
            .capability_snapshot()
            .concurrency
            .transactional_results,
    };
    provider_resources.sort_by_key(|resource| resource.device_id);
    let digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &provider_resources))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: provider_resources,
        epoch: u64::from_le_bytes(digest.bytes()[..8].try_into().expect("digest length")),
    };
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::ProviderRequired {
            device_id: provider.device_id(),
        },
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(request.semantic_operation)
    {
        return Err(
            build_runtime_error("placement selected an unadmitted provider operation")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(query)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(provider.device_id(), query.operation.clone());
    if request.conditional_host_fallback.is_some() {
        receipt.allow_conditional_host_fallback();
    }
    if conditional_provider_upload.is_some() {
        receipt.allow_conditional_provider_upload();
    }
    Ok(Some(SelectedProviderOperation {
        provider,
        receipt: HostPresentationReceipt::new(runtime.clone(), selected, receipt),
        conditional_output_requested_bytes: conditional_provider_upload
            .map(|upload| upload.provider_output_requested_bytes),
    }))
}

fn sum_resource_bounds(
    left: ResourceByteBound,
    right: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => left
            .checked_add(right)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error("provider input stage byte bound overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}

fn peak_resource_bounds(left: ResourceByteBound, right: ResourceByteBound) -> ResourceByteBound {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
            ResourceByteBound::Known(left.max(right))
        }
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => ResourceByteBound::Unavailable(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_input_stage_overflow_is_a_resource_error_not_a_missing_selection() {
        let error = sum_resource_bounds(
            ResourceByteBound::Known(u64::MAX),
            ResourceByteBound::Known(1),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:ResourceOverflow")
        );
    }

    #[test]
    fn provider_input_stage_unknown_never_becomes_zero() {
        let unknown = ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert_eq!(sum_resource_bounds(16.into(), unknown).unwrap(), unknown);
    }

    #[test]
    fn conditional_host_peak_is_maximum_of_sequential_branches() {
        assert_eq!(peak_resource_bounds(24.into(), 32.into()), 32.into());
        assert_eq!(peak_resource_bounds(40.into(), 32.into()), 40.into());
        let unknown = ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert_eq!(peak_resource_bounds(40.into(), unknown), unknown);
    }
}
