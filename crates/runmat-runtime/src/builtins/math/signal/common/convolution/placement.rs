//! Whole-call placement for the two live-handle convolution hooks. Public
//! parsing and semantic admission remain with the identity owners.

use runmat_accelerate_api::{
    same_provider_instance, GpuTensorHandle, ProviderConvolutionForm,
    ProviderConvolutionInputBackings, ProviderConvolutionWorkload, ProviderElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkObservation, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use runmat_time::{duration_ns_saturating, Instant};
use runmat_types::ConvolutionMode;

use crate::context::{RuntimeContext, RuntimeWholeWorkSelection};
use crate::BuiltinResult;

/// The receipt is tied to the actual borrowed pair, not a serialized shape.
/// Dropping a pending call releases its exploration probe without feedback.
pub(crate) struct SelectedConvolution {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    started: Instant,
    provider_route: bool,
    backing: ProviderConvolutionInputBackings,
}

impl SelectedConvolution {
    pub(crate) fn provider_route(&self) -> bool {
        self.provider_route
    }

    pub(crate) fn revalidate(&self, handles: [&GpuTensorHandle; 2]) -> BuiltinResult<bool> {
        self.backing.matches_live_handles(handles).map_err(|cause| {
            crate::build_runtime_error("convolution input lease expired before dispatch")
                .with_identifier("RunMat:Placement:ExpiredConvolutionInput")
                .with_source(cause)
                .build()
        })
    }

    pub(crate) fn finish_hook(self, outcome: &BuiltinResult<Option<runmat_value::Value>>) {
        self.finish(outcome.is_ok(), outcome.as_ref().err());
    }

    pub(crate) fn finish_value(self, outcome: &BuiltinResult<runmat_value::Value>) {
        let published_on_provider = matches!(outcome, Ok(runmat_value::Value::GpuTensor(_)));
        self.finish(published_on_provider, outcome.as_ref().err());
    }

    fn finish(self, published: bool, error: Option<&crate::RuntimeError>) {
        let status = match error {
            None if published => WholeWorkObservationOutcome::Success,
            None => WholeWorkObservationOutcome::Abandoned,
            Some(error)
                if self.runtime.is_cancelled()
                    || error.identifier() == Some("RunMat:Cancelled") =>
            {
                WholeWorkObservationOutcome::Cancelled
            }
            Some(_) => WholeWorkObservationOutcome::OperationalFailure,
        };
        let observation = WholeWorkObservation {
            signature: self.selected.decision.signature.clone(),
            selected: self.selected.decision.selected.clone(),
            elapsed_ns: duration_ns_saturating(self.started.elapsed()),
            outcome: status,
        };
        if let Some(service) = self.runtime.service_ports().placement() {
            if let Err(error) = service.observe_whole_work(observation) {
                tracing::warn!(target: "runmat_placement", "convolution observation rejected: {error}");
            }
        }
    }
}

fn unknown_scratch() -> ResourceByteBound {
    ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
}

fn representation(
    handle: &GpuTensorHandle,
    element_type: ProviderElementType,
) -> Option<ProviderRepresentation> {
    let shape = handle
        .shape()
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    Some(ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    })
}

/// None keeps the pre-existing route when no whole-work authority exists or
/// when this exact pair is not a physical provider-hook candidate. A rejected
/// provider query still permits the admitted host materialization route.
pub(crate) fn select_pair(
    runtime: Option<&RuntimeContext>,
    operation: &'static str,
    handles: [&GpuTensorHandle; 2],
    mode: ConvolutionMode,
    form: ProviderConvolutionForm,
    host_result_bytes: Option<u64>,
) -> BuiltinResult<Option<SelectedConvolution>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handles[0]) else {
        return Ok(None);
    };
    let Some(right_provider) = runmat_accelerate_api::provider_for_handle(handles[1]) else {
        return Ok(None);
    };
    if !same_provider_instance(provider, right_provider)
        || handles[0].device_id() != handles[1].device_id()
    {
        return Ok(None);
    }
    let backing = match ProviderConvolutionInputBackings::from_full_live_handles(handles) {
        Ok(Some(backing)) => backing,
        Ok(None) => return Ok(None),
        Err(cause) => {
            return Err(
                crate::build_runtime_error("convolution input lease is not live")
                    .with_identifier("RunMat:Placement:ExpiredConvolutionInput")
                    .with_source(cause)
                    .build(),
            )
        }
    };
    let Some(precision) = runmat_accelerate_api::handle_precision(handles[0]) else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_precision(handles[1]) != Some(precision)
        || runmat_accelerate_api::handle_storage(handles[0])
            != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(handles[1])
            != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(None);
    }
    let element = ProviderElementType::from(precision);
    let Some(left) = representation(handles[0], element) else {
        return Ok(None);
    };
    let Some(right) = representation(handles[1], element) else {
        return Ok(None);
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(operation),
        family: ProviderOperationFamily::Library,
        inputs: vec![left, right],
        outputs: Vec::new(),
        workload: ProviderWorkload {
            convolution: Some(ProviderConvolutionWorkload {
                form,
                mode,
                input_backings: Some(backing),
            }),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    let feasibility = provider.query_feasibility(&query);
    let capability = provider.capability_snapshot();
    let provider_digest =
        Digest::sha256(serde_json::to_vec(&capability).expect("provider capability serializes"));
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let signature = PlacementSignature {
        region: None,
        operation: operation.to_owned(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                &query,
                provider.device_id(),
                handles.map(runmat_accelerate_api::handle_provenance),
            ))
            .expect("typed convolution query serializes"),
        ),
        revision,
    };
    let device_id = provider.device_id();
    let host = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Provider { device_id },
        presentation: CandidateOutputPresentation::ProviderUpload {
            device_id,
            transfer_bytes: host_result_bytes,
            host_scratch_bytes: unknown_scratch(),
            provider_scratch_bytes: unknown_scratch(),
            queue_slots: 1,
            preparation: CandidatePreparationState::Ready,
        },
        route_facts: Digest::sha256(b"runmat.signal.convolution.host_gather_compute_restore.v1"),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: host_result_bytes.map(ResourceByteBound::Known).unwrap_or(
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
            ),
            scratch_bytes: unknown_scratch(),
            queue_slots: 0,
        },
        transactional_results: true,
    };
    let mut candidates = vec![host];
    if let ProviderFeasibility::Supported { estimate } = feasibility {
        candidates.push(WholeWorkCandidate {
            kind: ExecutionCandidateKind::ProviderOperation,
            execution_location: CandidateExecutionLocation::Provider { device_id },
            preparation: CandidatePreparationState::Unknown,
            output_residency: CandidateOutputResidency::Provider { device_id },
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(
                serde_json::to_vec(&(query.clone(), estimate.clone()))
                    .expect("typed provider route serializes"),
            ),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 0,
                retained_bytes: estimate
                    .output_bytes
                    .map(ResourceByteBound::Known)
                    .unwrap_or(ResourceByteBound::Unavailable(
                        ResourceBoundUnavailable::NoRetainedValueBound,
                    )),
                // A peak-live bound is not a transient-scratch bound.
                scratch_bytes: unknown_scratch(),
                queue_slots: estimate.dispatches.unwrap_or(1),
            },
            transactional_results: capability.concurrency.transactional_results,
        });
    }
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let provider_resource = provider.placement_resources();
    let epoch_bytes = serde_json::to_vec(&(parallel.epoch, &provider_resource))
        .expect("resource snapshot serializes");
    let digest = Digest::sha256(epoch_bytes);
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: vec![provider_resource],
        epoch: u64::from_le_bytes(digest.bytes()[..8].try_into().expect("digest length")),
    };
    let chosen = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: candidates.clone(),
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if chosen.decision.signature != signature {
        return Err(
            crate::build_runtime_error("placement selected a different convolution")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let Some(index) = candidates
        .iter()
        .position(|candidate| candidate.selected(operation) == chosen.decision.selected)
    else {
        return Err(crate::build_runtime_error(
            "placement selected an unadmitted convolution route",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .build());
    };
    Ok(Some(SelectedConvolution {
        runtime: runtime.clone(),
        selected: chosen,
        started: Instant::now(),
        provider_route: index == 1,
        backing,
    }))
}
