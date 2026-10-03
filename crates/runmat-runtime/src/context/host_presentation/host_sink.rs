//! Whole-call host publication after an awaited resident readback. The
//! receipt times preparation, download, host computation and publication;
//! no upload or provider dispatch is asserted for this route.

use super::admission::combine_host_scratch;
use crate::context::{RuntimeContext, RuntimeWholeWorkSelection};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    provider_for_handle, GpuTensorHandle, NumericElementType, ProviderElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementFeedback,
    PlacementResourceSnapshot, PlacementSignature, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkPlacementRequest,
};
use std::time::Instant;

pub(crate) struct ExactHostSink<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) source: &'a GpuTensorHandle,
    pub(crate) domain_facts: Digest,
    pub(crate) output_bytes: ResourceByteBound,
    /// Operation-owned host temporaries, excluding the provider's native
    /// returned payload and provider-owned mapped/staging allocations.
    pub(crate) host_compute_scratch: ResourceByteBound,
}

pub(crate) struct HostSinkReceipt {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    started: Instant,
}

impl HostSinkReceipt {
    /// Call only after the awaited readback and final host value are complete.
    /// Dropping before this point abandons the sample and probe lease.
    pub(crate) fn finish(self, published: bool) {
        if self.runtime.is_cancelled() {
            return;
        }
        if let Some(service) = self.runtime.service_ports().placement() {
            let _ = service.observe(PlacementFeedback {
                signature: self.selected.decision.signature.clone(),
                candidate: self.selected.decision.selected.candidate.clone(),
                kind: self.selected.decision.selected.kind,
                total_elapsed_ns: runmat_time::duration_ns_saturating(self.started.elapsed())
                    .max(1),
                succeeded: published,
            });
        }
    }
}

pub(crate) fn select_exact_host_sink(
    runtime: Option<&RuntimeContext>,
    request: ExactHostSink<'_>,
) -> BuiltinResult<Option<HostSinkReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(provider) = provider_for_handle(request.source) else {
        return Ok(None);
    };
    if provider.device_id() != request.source.device_id()
        || request.source.allocation_lease().validate_live().is_err()
    {
        return Err(
            build_runtime_error("host sink source has no live allocation owner")
                .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                .build(),
        );
    }
    let input = ProviderRepresentation {
        element_type: match request.source.physical().element_type() {
            NumericElementType::F32 => ProviderElementType::F32,
            NumericElementType::F64 => ProviderElementType::F64,
            NumericElementType::I8 => ProviderElementType::I8,
            NumericElementType::I16 => ProviderElementType::I16,
            NumericElementType::I32 => ProviderElementType::I32,
            NumericElementType::I64 => ProviderElementType::I64,
            NumericElementType::U8 => ProviderElementType::U8,
            NumericElementType::U16 => ProviderElementType::U16,
            NumericElementType::U32 => ProviderElementType::U32,
            NumericElementType::U64 => ProviderElementType::U64,
        },
        storage: ProviderStorage::DenseReal,
        layout: if runmat_accelerate_api::handle_is_transposed(request.source) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: request
            .source
            .shape()
            .iter()
            .map(|&extent| u64::try_from(extent).ok())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                build_runtime_error("host sink input geometry overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            })?,
        residency: ProviderResidency::Device,
    };
    let mut output = input.clone();
    output.residency = ProviderResidency::Host;
    // A lazy source is read through its strided view, but the completed host
    // payload is native column-major data in the logical output shape.
    output.layout = ProviderLayout::ColumnMajorContiguous;
    let download = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.download"),
        family: ProviderOperationFamily::Download,
        inputs: vec![input],
        outputs: vec![output],
        workload: ProviderWorkload {
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let provider_resource = provider.placement_resources();
    if !matches!(
        provider.query_numeric_download_feasibility(request.source, &download),
        ProviderFeasibility::Supported { .. }
    ) {
        if parallel.memory_available_bytes.is_some()
            || provider_resource.capacity_bytes.is_some()
            || provider_resource.scratch_available_bytes.is_some()
        {
            return Err(build_runtime_error(
                "host sink has no proven download route under a finite resource budget",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .build());
        }
        return Ok(None);
    }
    let readback_host = provider.download_numeric_host_workspace_bound(request.source);
    let host_scratch = combine_host_scratch(request.host_compute_scratch, readback_host)?;
    let provider_scratch = provider.download_numeric_provider_scratch_bound(request.source);
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&provider.capability_snapshot())
            .expect("provider capability serializes"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let signature = PlacementSignature {
        region: None,
        operation: request.operation.into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                request.variant,
                request.domain_facts,
                &download,
                request.output_bytes,
                readback_host,
                provider_scratch,
                provider.device_id(),
                request.source.view().layout(),
                runmat_accelerate_api::handle_provenance(request.source),
            ))
            .expect("typed host-sink facts serialize"),
        ),
        revision,
    };
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Host,
        presentation: CandidateOutputPresentation::ProviderDownload {
            device_id: provider.device_id(),
            provider_scratch_bytes: provider_scratch,
            queue_slots: 1,
        },
        route_facts: Digest::sha256(
            serde_json::to_vec(&download).expect("download route serializes"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: request.output_bytes,
            scratch_bytes: host_scratch,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    };
    let epoch_digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &provider_resource))
            .expect("resource snapshot serializes"),
    );
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: vec![provider_resource],
        epoch: u64::from_le_bytes(epoch_digest.bytes()[..8].try_into().expect("digest length")),
    };
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature: signature.clone(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if selected.decision.signature != signature
        || selected.decision.selected != candidate.selected(request.operation)
    {
        return Err(
            build_runtime_error("placement selected an unadmitted host sink route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    Ok(Some(HostSinkReceipt {
        runtime: runtime.clone(),
        selected,
        started: Instant::now(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_presentation::tests::Witness;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::{
        CancellationReason, PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind,
    };
    use std::rc::Rc;

    fn selected() -> RuntimeWholeWorkSelection {
        let digest = Digest::sha256(b"host-sink-cancellation");
        RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: PlacementSignature {
                    region: None,
                    operation: "gray2rgb".into(),
                    runtime_facts: digest,
                    revision: PlacementRevision {
                        program: None,
                        catalog: digest,
                        compiler: digest,
                        provider: digest,
                        policy: digest,
                    },
                },
                selected: WholeWorkCandidate {
                    kind: ExecutionCandidateKind::SharedRuntime,
                    execution_location: CandidateExecutionLocation::Host,
                    preparation: CandidatePreparationState::Ready,
                    output_residency: CandidateOutputResidency::Host,
                    presentation: CandidateOutputPresentation::Direct,
                    route_facts: digest,
                    cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
                    resources: CandidateResourceDemand::default(),
                    transactional_results: true,
                }
                .selected("gray2rgb"),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
        }
    }

    #[test]
    fn cancellation_and_abandoned_host_sink_never_publish_feedback() {
        let witness = Rc::new(Witness::default());
        let cancellation = CancellationSource::new();
        let runtime = witness
            .context()
            .with_cancellation_token(cancellation.token());
        drop(HostSinkReceipt {
            runtime: runtime.clone(),
            selected: selected(),
            started: Instant::now(),
        });
        assert!(witness.feedback().is_empty());
        cancellation.cancel(CancellationReason::User);
        HostSinkReceipt {
            runtime,
            selected: selected(),
            started: Instant::now(),
        }
        .finish(true);
        assert!(witness.feedback().is_empty());
    }
}
