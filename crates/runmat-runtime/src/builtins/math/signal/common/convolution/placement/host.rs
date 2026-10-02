//! Exact and bounded host-produced provider-presentation admission.
use super::pair::unknown_scratch;
use super::*;

/// A host-only convolution variant can still be admitted as a whole call when
/// its published payload has an authenticated owner and a feasible, observable
/// upload. The typed output list covers separable and multi-output variants;
/// unsupported classes retain their existing execution path unchanged.
pub(crate) fn select_host_variant(
    runtime: Option<&RuntimeContext>,
    operation: &'static str,
    variant: &'static str,
    mode: Option<ConvolutionMode>,
    inputs: &[&Value],
    resident_inputs: &[GpuTensorHandle],
    outputs: Vec<ProviderRepresentation>,
    host_compute_scratch: ResourceByteBound,
) -> BuiltinResult<Option<SelectedConvolution>> {
    select_host_admission(
        runtime,
        operation,
        variant,
        mode,
        inputs,
        resident_inputs,
        outputs,
        None,
        host_compute_scratch,
    )
}

pub(crate) fn select_bounded_host_variant(
    runtime: Option<&RuntimeContext>,
    operation: &'static str,
    variant: &'static str,
    inputs: &[&Value],
    resident_inputs: &[GpuTensorHandle],
    bounds: Vec<ProviderTransferOutputBound>,
    host_compute_scratch: ResourceByteBound,
) -> BuiltinResult<Option<SelectedConvolution>> {
    select_host_admission(
        runtime,
        operation,
        variant,
        None,
        inputs,
        resident_inputs,
        Vec::new(),
        Some(bounds),
        host_compute_scratch,
    )
}

fn select_host_admission(
    runtime: Option<&RuntimeContext>,
    operation: &'static str,
    variant: &'static str,
    mode: Option<ConvolutionMode>,
    inputs: &[&Value],
    resident_inputs: &[GpuTensorHandle],
    outputs: Vec<ProviderRepresentation>,
    bounds: Option<Vec<ProviderTransferOutputBound>>,
    host_compute_scratch: ResourceByteBound,
) -> BuiltinResult<Option<SelectedConvolution>> {
    let (Some(runtime), Some(owner)) = (runtime, resident_inputs.first()) else {
        return Ok(None);
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(None);
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(owner) else {
        return Ok(None);
    };
    if resident_inputs.iter().any(|handle| {
        handle.device_id() != owner.device_id()
            || !runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|candidate| same_provider_instance(candidate, provider))
    }) {
        // Cross-owner gathers are legal in the legacy host implementation,
        // but this one-provider resource snapshot does not describe them.
        return Ok(None);
    }
    if (outputs.is_empty() && bounds.as_ref().is_none_or(Vec::is_empty))
        || outputs.iter().any(|output| {
            output.residency != ProviderResidency::Device || output.checked_byte_len().is_none()
        })
    {
        return Ok(None);
    }
    let transfer_bytes = if let Some(bounds) = &bounds {
        bounds
            .iter()
            .try_fold(0_u64, |total, bound| total.checked_add(bound.max_bytes()?))
    } else {
        outputs.iter().try_fold(0_u64, |total, output| {
            total.checked_add(output.checked_byte_len()?)
        })
    };
    let Some(transfer_bytes) = transfer_bytes else {
        return Ok(None);
    };
    let upload = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("transfer.upload"),
        family: ProviderOperationFamily::Upload,
        inputs: outputs
            .iter()
            .cloned()
            .map(|mut input| {
                input.residency = ProviderResidency::Host;
                input
            })
            .collect(),
        outputs: outputs.clone(),
        workload: ProviderWorkload {
            requested_outputs: u32::try_from(bounds.as_ref().map_or(outputs.len(), Vec::len)).ok(),
            transfer_output_bounds: bounds.clone(),
            ..ProviderWorkload::default()
        },
    };
    if !provider.query_feasibility(&upload).is_supported()
        || !matches!(
            provider.operation_observation_capability(&upload),
            runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_)
        )
    {
        return Ok(None);
    }
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&provider.capability_snapshot())
            .expect("provider capability serializes"),
    );
    let Some(revision) =
        service.whole_work_revision(runtime.program_revision().cloned(), provider_digest)
    else {
        return Ok(None);
    };
    let operand_facts = inputs
        .iter()
        .map(|value| input_facts(value))
        .collect::<Vec<_>>();
    let signature = PlacementSignature {
        region: None,
        operation: operation.to_owned(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&(
                variant,
                mode,
                &operand_facts,
                &outputs,
                &bounds,
                resident_inputs
                    .iter()
                    .map(|handle| {
                        (
                            handle.device_id(),
                            runmat_accelerate_api::handle_provenance(handle),
                        )
                    })
                    .collect::<Vec<_>>(),
            ))
            .expect("typed host convolution route serializes"),
        ),
        revision,
    };
    let device_id = provider.device_id();
    let candidate = WholeWorkCandidate {
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Provider { device_id },
        presentation: CandidateOutputPresentation::ProviderUpload {
            device_id,
            transfer_bytes: Some(transfer_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(transfer_bytes)),
            host_scratch_bytes: ResourceByteBound::Known(transfer_bytes),
            provider_scratch_bytes: unknown_scratch(),
            queue_slots: u32::try_from(bounds.as_ref().map_or(outputs.len(), Vec::len))
                .unwrap_or(u32::MAX),
            preparation: CandidatePreparationState::Ready,
        },
        route_facts: Digest::sha256(
            serde_json::to_vec(&(variant, mode, &upload)).expect("typed route serializes"),
        ),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            retained_bytes: ResourceByteBound::Known(transfer_bytes),
            scratch_bytes: host_compute_scratch,
            queue_slots: 0,
            peak_live: None,
        },
        transactional_results: true,
    };
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|port| port.placement_resources())
        .unwrap_or_default();
    let provider_resource = provider.placement_resources();
    let digest = Digest::sha256(
        serde_json::to_vec(&(parallel.epoch, &provider_resource))
            .expect("resource snapshot serializes"),
    );
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
        candidates: vec![candidate.clone()],
        resources,
        deterministic: true,
        require_transactional_results: false,
    })?;
    if chosen.decision.signature != signature
        || chosen.decision.selected != candidate.selected(operation)
    {
        return Err(crate::build_runtime_error(
            "placement selected an unadmitted host convolution route",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .build());
    }
    let ProviderObservationSupport::Observed(mut receipt) =
        provider.begin_operation_observation(&upload)
    else {
        return Ok(None);
    };
    receipt.require_provider_execution(device_id, upload.operation);
    Ok(Some(SelectedConvolution {
        runtime: runtime.clone(),
        selected: chosen,
        provider_route: false,
        backing: None,
        receipt: Some(receipt),
        expected_outputs: outputs,
        expected_bounds: bounds,
        expected_device: device_id,
    }))
}

fn input_facts(value: &Value) -> (String, Vec<usize>) {
    match value {
        Value::GpuTensor(handle) => (
            format!(
                "gpu:{:?}:{:?}:{:?}:{:?}:{}",
                runmat_accelerate_api::handle_precision(handle),
                runmat_accelerate_api::handle_integer_type(handle),
                runmat_accelerate_api::handle_storage(handle),
                runmat_accelerate_api::handle_provenance(handle),
                handle.device_id(),
            ),
            handle.shape().to_vec(),
        ),
        Value::Tensor(tensor) => (
            format!(
                "tensor:{:?}:{:?}",
                tensor.numeric_dtype(),
                tensor.integer_storage()
            ),
            tensor.shape.clone(),
        ),
        Value::ComplexTensor(tensor) => (
            format!("complex:{:?}", tensor.numeric_dtype()),
            tensor.shape.clone(),
        ),
        Value::LogicalArray(array) => ("logical".into(), array.shape.clone()),
        Value::Num(_) => ("double_scalar".into(), vec![1, 1]),
        Value::Int(value) => (format!("integer_scalar:{value:?}"), vec![1, 1]),
        Value::Bool(_) => ("logical_scalar".into(), vec![1, 1]),
        Value::Complex(_, _) => ("complex_scalar".into(), vec![1, 1]),
        _ => ("unsupported".into(), vec![]),
    }
}

pub(crate) fn host_output_representations(
    shapes: &[Vec<usize>],
    single: bool,
    complex: bool,
) -> Option<Vec<ProviderRepresentation>> {
    let element_type = match (single, complex) {
        (true, false) => ProviderElementType::F32,
        (false, false) => ProviderElementType::F64,
        (true, true) => ProviderElementType::ComplexF32,
        (false, true) => ProviderElementType::ComplexF64,
    };
    let storage = if complex {
        ProviderStorage::DenseComplexInterleaved
    } else {
        ProviderStorage::DenseReal
    };
    shapes
        .iter()
        .map(|shape| {
            Some(ProviderRepresentation {
                element_type,
                storage,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: shape
                    .iter()
                    .copied()
                    .map(u64::try_from)
                    .collect::<Result<_, _>>()
                    .ok()?,
                residency: ProviderResidency::Device,
            })
        })
        .collect()
}
