//! Shared exact host-result upload admission. The caller supplies checked
//! operation-owned geometry and scratch; this boundary asks the owner about
//! its readback host workspace and actual upload route, then submits one
//! whole-work candidate to policy.

use super::HostPresentationReceipt;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    provider_for_handle, GpuTensorHandle, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ProviderReadbackDemand, ProviderReadbackDemands, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkCandidate, WholeWorkConstraint, WholeWorkPlacementRequest,
};

pub(crate) struct ExactHostUpload<'a> {
    pub(crate) operation: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) owner: &'a GpuTensorHandle,
    /// Digest of typed input/control/output-mode facts, never a serialized
    /// substitute for the live owner handle used in provider admission.
    pub(crate) domain_facts: Digest,
    pub(crate) outputs: Vec<ProviderRepresentation>,
    /// Host temporaries owned by the operation, excluding source readback
    /// payload and provider-internal readback workspace. Shared admission
    /// queries the owner for those before submitting a host-memory bound.
    pub(crate) host_compute_scratch: ResourceByteBound,
}

pub(crate) fn select_exact_host_upload(
    runtime: Option<&RuntimeContext>,
    request: ExactHostUpload<'_>,
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let owner = request.owner;
    select_exact_host_upload_with_readbacks(runtime, request, std::slice::from_ref(owner))
}

/// The readback set is exactly the resident operands that the host computation
/// will gather. Each independently returned native payload (including aliases)
/// contributes to peak host workspace. The owner remains the original handle
/// selected for result publication, not an automatically prepared replacement.
pub(crate) fn select_exact_host_upload_with_readbacks(
    runtime: Option<&RuntimeContext>,
    request: ExactHostUpload<'_>,
    readbacks: &[GpuTensorHandle],
) -> BuiltinResult<Option<HostPresentationReceipt>> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(provider) = provider_for_handle(request.owner) else {
        return Ok(None);
    };
    // A binary host conversion may gather from two independently owned
    // devices. Only the final publication is constrained to `request.owner`.
    let source_providers = readbacks
        .iter()
        .map(|source| {
            provider_for_handle(source).filter(|source_provider| {
                source.device_id() == source_provider.device_id()
                    && source.allocation_lease().validate_live().is_ok()
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(source_providers) = source_providers.filter(|providers| !providers.is_empty()) else {
        return Err(
            build_runtime_error("host readback source has no live allocation owner")
                .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                .build(),
        );
    };
    if request.outputs.is_empty()
        || request.outputs.iter().any(|output| {
            output.residency != ProviderResidency::Device || output.checked_byte_len().is_none()
        })
    {
        return Ok(None);
    }
    let Some(transfer_bytes) = request.outputs.iter().try_fold(0_u64, |total, output| {
        total.checked_add(output.checked_byte_len()?)
    }) else {
        return Ok(None);
    };
    let requested_outputs = u32::try_from(request.outputs.len()).ok();
    let upload = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: request
            .outputs
            .iter()
            .cloned()
            .map(|mut input| {
                input.residency = ProviderResidency::Host;
                input
            })
            .collect(),
        outputs: request.outputs,
        workload: ProviderWorkload {
            requested_outputs,
            ..ProviderWorkload::default()
        },
    };
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&upload) else {
        return Ok(None);
    };
    if !matches!(
        provider.operation_observation_capability(&upload),
        ProviderObservationCapability::Supported(_)
    ) {
        return Ok(None);
    }
    let readback_host_workspace = readbacks.iter().zip(&source_providers).try_fold(
        ResourceByteBound::Known(0),
        |total, (source, source_provider)| {
            combine_host_scratch(
                total,
                source_provider.download_numeric_host_workspace_bound(source),
            )
        },
    )?;
    let total_host_scratch =
        combine_host_scratch(request.host_compute_scratch, readback_host_workspace)?;
    // The gathers complete sequentially. Per owner, their peak staging and
    // occupied queue slot are therefore maxima, not a sum of unrelated
    // simultaneously live submissions. Unknown provider staging stays unknown.
    let readback_demands =
        ProviderReadbackDemands::sequential(readbacks.iter().zip(&source_providers).map(
            |(source, source_provider)| {
                let provider_scratch_bytes =
                    source_provider.download_numeric_provider_scratch_bound(source);
                ProviderReadbackDemand {
                    device_id: source_provider.device_id(),
                    provider_scratch_bytes,
                    queue_slots: u32::from(
                        !source.shape().contains(&0)
                            || provider_scratch_bytes != ResourceByteBound::Known(0),
                    ),
                }
            },
        ))
        .ok_or_else(|| {
            build_runtime_error("host readback exceeds the provider resource set")
                .with_identifier("RunMat:Placement:TooManyProviders")
                .build()
        })?;
    let device_id = provider.device_id();
    // The reusable decision must invalidate when any readback owner's
    // capability changes, not just when the upload owner's capability does.
    let source_capabilities = source_providers
        .iter()
        .map(|source_provider| {
            (
                source_provider.device_id(),
                source_provider.capability_snapshot(),
            )
        })
        .collect::<Vec<_>>();
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&(provider.capability_snapshot(), &source_capabilities))
            .expect("capability snapshots serialize"),
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
                &upload,
                readback_host_workspace,
                readback_demands,
                readbacks
                    .iter()
                    .zip(&source_providers)
                    .map(|(source, source_provider)| {
                        (
                            source_provider.device_id(),
                            source.shape(),
                            source.physical().element_type(),
                            source.physical().requested_bytes(),
                            source.view().layout(),
                            runmat_accelerate_api::handle_provenance(source),
                        )
                    })
                    .collect::<Vec<_>>(),
                device_id,
                runmat_accelerate_api::handle_provenance(request.owner),
            ))
            .expect("typed upload route serializes"),
        ),
        revision,
    };
    let candidate = upload_candidate(
        device_id,
        transfer_bytes,
        estimate.output_bytes,
        total_host_scratch,
        u32::try_from(upload.outputs.len()).unwrap_or(u32::MAX),
        Digest::sha256(serde_json::to_vec(&upload).expect("typed route serializes")),
        Some(readback_demands),
    );
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let mut provider_resources = vec![provider.placement_resources()];
    for source_provider in &source_providers {
        if !provider_resources
            .iter()
            .any(|resource| resource.device_id == source_provider.device_id())
        {
            provider_resources.push(source_provider.placement_resources());
        }
    }
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
            build_runtime_error("placement selected an unadmitted host upload route")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&upload)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(device_id, upload.operation);
    Ok(Some(HostPresentationReceipt::new(
        runtime.clone(),
        selected,
        receipt,
    )))
}

/// The caller's scratch excludes the provider's native readback payload and
/// its internal host temporaries. Unknown on either side remains unknown;
/// only a checked sum of proven bounds can pass a finite host-memory quota.
/// Overflow is a placement error, never an unselected route that could run
/// without admission.
pub(super) fn combine_host_scratch(
    operation_owned: ResourceByteBound,
    readback_host_workspace: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (operation_owned, readback_host_workspace) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => left
            .checked_add(right)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error("host readback workspace byte bound overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}

fn upload_candidate(
    device_id: u32,
    transfer_bytes: u64,
    physical_retained_bytes: Option<u64>,
    host_compute_scratch: ResourceByteBound,
    queue_slots: u32,
    route_facts: Digest,
    readbacks: Option<ProviderReadbackDemands>,
) -> WholeWorkCandidate {
    let presentation = match readbacks {
        Some(readbacks) => CandidateOutputPresentation::ProviderReadbackUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: ResourceByteBound::Known(transfer_bytes),
            provider_scratch_bytes: ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            ),
            queue_slots,
            preparation: CandidatePreparationState::Ready,
            readbacks,
        },
        None => CandidateOutputPresentation::ProviderUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: ResourceByteBound::Known(transfer_bytes),
            provider_scratch_bytes: ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            ),
            queue_slots,
            preparation: CandidatePreparationState::Ready,
        },
    };
    WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Provider { device_id },
        presentation,
        route_facts,
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            // Physical provider retention may include padding/alignment and
            // is not the host payload or transfer byte count. Unknown stays
            // unknown under a finite provider memory budget.
            retained_bytes: physical_retained_bound(physical_retained_bytes),
            scratch_bytes: host_compute_scratch,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    }
}

fn physical_retained_bound(output_bytes: Option<u64>) -> ResourceByteBound {
    output_bytes
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoRetainedValueBound,
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{PlacementRevision, ProviderResourceSnapshot};

    fn budget_request(logical: u64, physical: u64, capacity: u64) -> WholeWorkPlacementRequest {
        let digest = Digest::sha256(b"physical-host-upload-budget");
        let mut candidate = upload_candidate(
            17,
            logical,
            Some(physical),
            ResourceByteBound::Known(0),
            1,
            digest,
            None,
        );
        // This synthetic route isolates retained-provider and host-workspace
        // budgets. Unknown provider scratch would independently prevent its
        // acceptance under the finite provider capacity in this fixture.
        if let CandidateOutputPresentation::ProviderUpload {
            provider_scratch_bytes,
            ..
        } = &mut candidate.presentation
        {
            *provider_scratch_bytes = ResourceByteBound::Known(0);
        }
        WholeWorkPlacementRequest {
            signature: PlacementSignature {
                region: None,
                operation: "imhist".into(),
                runtime_facts: digest,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    provider: digest,
                    policy: digest,
                },
            },
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: vec![candidate],
            resources: PlacementResourceSnapshot {
                cpu_millicores_available: 1_000,
                memory_available_bytes: Some(1_024),
                cancellation_requested: false,
                providers: vec![ProviderResourceSnapshot {
                    device_id: 17,
                    capacity_bytes: Some(capacity),
                    scratch_available_bytes: None,
                    ..Default::default()
                }],
                epoch: 0,
            },
            deterministic: true,
            require_transactional_results: false,
        }
    }

    #[test]
    fn shared_service_budgets_physical_retention_not_logical_payload() {
        let session = runmat_accelerate::placement::PlacementSession::new(
            runmat_accelerate::placement::PlacementSessionConfig::default(),
        );
        assert!(session
            .select_whole_work(budget_request(32, 40, 39))
            .is_err());
        assert!(session
            .select_whole_work(budget_request(32, 40, 40))
            .is_ok());
        // Zero logical elements can still require a nonzero provider buffer.
        assert!(session.select_whole_work(budget_request(0, 8, 7)).is_err());
        assert!(session.select_whole_work(budget_request(0, 8, 8)).is_ok());
    }

    #[test]
    fn provider_retention_is_not_logical_transfer_or_unknown_as_zero() {
        let logical_transfer = 32;
        assert_eq!(
            physical_retained_bound(Some(40)),
            ResourceByteBound::Known(40)
        );
        assert_ne!(
            physical_retained_bound(Some(40)),
            ResourceByteBound::Known(logical_transfer)
        );
        // A zero-element logical payload can still retain a nonzero provider
        // allocation floor; never replace the reported physical result by 0.
        assert_eq!(
            physical_retained_bound(Some(8)),
            ResourceByteBound::Known(8)
        );
        assert_eq!(
            physical_retained_bound(None),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound)
        );
    }

    #[test]
    fn readback_workspace_combination_preserves_unknown_and_checks_sum() {
        assert_eq!(
            combine_host_scratch(ResourceByteBound::Known(56), ResourceByteBound::Known(16))
                .unwrap(),
            ResourceByteBound::Known(72)
        );
        assert_eq!(
            combine_host_scratch(
                ResourceByteBound::Known(56),
                ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            )
            .unwrap(),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
        );
        assert_eq!(
            combine_host_scratch(
                ResourceByteBound::Known(u64::MAX),
                ResourceByteBound::Known(1)
            )
            .unwrap_err()
            .identifier(),
            Some("RunMat:Placement:ResourceOverflow")
        );
    }

    #[test]
    fn unknown_readback_workspace_cannot_pass_finite_host_quota() {
        let session = runmat_accelerate::placement::PlacementSession::new(
            runmat_accelerate::placement::PlacementSessionConfig::default(),
        );
        let mut request = budget_request(32, 40, 1_024);
        request.candidates[0].resources.scratch_bytes =
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert!(session.select_whole_work(request.clone()).is_err());
        request.resources.memory_available_bytes = None;
        assert!(session.select_whole_work(request).is_ok());
    }

    #[test]
    fn known_readback_workspace_is_charged_to_host_peak() {
        let session = runmat_accelerate::placement::PlacementSession::new(
            runmat_accelerate::placement::PlacementSessionConfig::default(),
        );
        let mut request = budget_request(32, 40, 1_024);
        // 72 readback + operation scratch, 32 upload staging, 32 live result.
        request.candidates[0].resources.scratch_bytes = ResourceByteBound::Known(72);
        request.resources.memory_available_bytes = Some(135);
        assert!(session.select_whole_work(request.clone()).is_err());
        request.resources.memory_available_bytes = Some(136);
        assert!(session.select_whole_work(request).is_ok());
    }
}
