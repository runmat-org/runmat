//! Admission of paired live-handle provider and host convolution routes.
use super::*;

pub(super) fn unknown_scratch() -> ResourceByteBound {
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
    host_output_shape: &[usize],
    host_compute_scratch: ResourceByteBound,
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
    let Some(output_shape) = host_output_shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
    else {
        return Ok(None);
    };
    let output = ProviderRepresentation {
        element_type: element,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: output_shape,
        residency: ProviderResidency::Device,
    };
    let host_result_bytes = output.checked_byte_len();
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(operation),
        family: ProviderOperationFamily::Library,
        inputs: vec![left, right],
        outputs: vec![output.clone()],
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
    // A provider without complete-call evidence keeps the existing direct
    // execution route; enqueue alone cannot train a placement profile.
    if !matches!(
        provider.operation_observation_capability(&query),
        runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_)
    ) {
        return Ok(None);
    }
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
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let device_id = provider.device_id();
    let mut candidates = Vec::new();
    let host_upload = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: vec![ProviderRepresentation {
            residency: ProviderResidency::Host,
            ..output.clone()
        }],
        outputs: vec![output.clone()],
        workload: ProviderWorkload::default(),
    };
    if provider.query_feasibility(&host_upload).is_supported()
        && matches!(
            provider.operation_observation_capability(&host_upload),
            runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_)
        )
    {
        candidates.push(WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Provider { device_id },
            presentation: CandidateOutputPresentation::ProviderUpload {
                device_id,
                transfer_bytes: host_result_bytes,
                host_result_bytes: host_result_bytes.map(ResourceByteBound::Known),
                host_scratch_bytes: host_result_bytes
                    .map(ResourceByteBound::Known)
                    .unwrap_or(unknown_scratch()),
                provider_scratch_bytes: unknown_scratch(),
                queue_slots: 1,
                preparation: CandidatePreparationState::Ready,
            },
            route_facts: Digest::sha256(
                b"runmat.signal.convolution.host_gather_compute_restore.v2",
            ),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: host_result_bytes.map(ResourceByteBound::Known).unwrap_or(
                    ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
                ),
                scratch_bytes: host_compute_scratch,
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: 0,
                peak_live: None,
            },
            transactional_results: true,
        });
    }
    if let ProviderFeasibility::Supported { estimate } = feasibility {
        let unique_live = backing.unique_live_bytes(handles).map_err(|cause| {
            crate::build_runtime_error("convolution input lease changed during placement")
                .with_identifier("RunMat:Placement:ExpiredConvolutionInput")
                .with_source(cause)
                .build()
        })?;
        let peak_live =
            estimate
                .transient_bytes
                .zip(unique_live)
                .map(|(peak_bytes, already_live_bytes)| {
                    runmat_execution::CandidatePeakLiveDemand {
                        peak_bytes,
                        already_live_bytes,
                    }
                });
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
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: estimate.dispatches.unwrap_or(1),
                peak_live,
            },
            transactional_results: capability.concurrency.transactional_results,
        });
    }
    if candidates.is_empty() {
        return Ok(None);
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
    let provider_route = candidates[index].kind.is_provider();
    let observed_query = if provider_route { &query } else { &host_upload };
    let receipt = {
        match provider.begin_operation_observation(observed_query) {
            ProviderObservationSupport::Observed(mut receipt) => {
                receipt.require_provider_execution(device_id, observed_query.operation.clone());
                Some(receipt)
            }
            ProviderObservationSupport::Unsupported => return Ok(None),
        }
    };
    Ok(Some(SelectedConvolution {
        runtime: runtime.clone(),
        selected: chosen,
        provider_route,
        backing: Some(backing),
        receipt,
        expected_outputs: vec![output],
        expected_bounds: None,
        expected_device: device_id,
    }))
}
