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
        bounds.iter().try_fold(0_u64, |total, bound| {
            // Transfer volume is logical payload. A provider may reserve a
            // nonzero physical buffer for an empty output; use its estimate
            // below for retained device bytes, not for copy volume.
            let bytes = bound.max_bytes()?;
            total.checked_add(if bound.max_elements == 0 { 0 } else { bytes })
        })
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
    let physical_retained_bytes = match provider.query_feasibility(&upload) {
        ProviderFeasibility::Supported { estimate } => estimate.output_bytes,
        ProviderFeasibility::Rejected { .. } => None,
    };
    let Some(physical_retained_bytes) = physical_retained_bytes else {
        return Ok(None);
    };
    if !matches!(
        provider.operation_observation_capability(&upload),
        runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_)
    ) {
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
        provider_provenance: None,
        graph_output_authority: None,
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
            retained_bytes: ResourceByteBound::Known(physical_retained_bytes),
            scratch_bytes: host_compute_scratch,
            provider_host_workspace: None,
            provider_input_stages: None,
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

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
struct InputFacts {
    class: InputClass,
    shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
enum InputClass {
    Resident {
        precision: Option<runmat_accelerate_api::ProviderPrecision>,
        integer: Option<runmat_accelerate_api::IntegerElementType>,
        storage: runmat_accelerate_api::GpuTensorStorage,
        provenance: Option<runmat_accelerate_api::GpuHandleProvenance>,
        logical: bool,
        device: u32,
    },
    Dense(runmat_value::NumericDType),
    ComplexDense(runmat_value::NumericDType),
    Logical,
    DoubleScalar,
    IntegerScalar(runmat_value::NumericDType),
    LogicalScalar,
    ComplexScalar,
    Unsupported,
}

fn input_facts(value: &Value) -> InputFacts {
    match value {
        Value::GpuTensor(handle) => InputFacts {
            class: InputClass::Resident {
                precision: runmat_accelerate_api::handle_precision(handle),
                integer: runmat_accelerate_api::handle_integer_type(handle),
                storage: runmat_accelerate_api::handle_storage(handle),
                provenance: runmat_accelerate_api::handle_provenance(handle),
                logical: runmat_accelerate_api::handle_is_logical(handle),
                device: handle.device_id(),
            },
            shape: handle.shape().to_vec(),
        },
        Value::Tensor(tensor) => InputFacts {
            class: InputClass::Dense(tensor.numeric_dtype()),
            shape: tensor.shape().to_vec(),
        },
        Value::ComplexTensor(tensor) => InputFacts {
            class: InputClass::ComplexDense(tensor.numeric_dtype()),
            shape: tensor.shape().to_vec(),
        },
        Value::LogicalArray(array) => InputFacts {
            class: InputClass::Logical,
            shape: array.shape.to_vec(),
        },
        Value::Num(_) => InputFacts {
            class: InputClass::DoubleScalar,
            shape: vec![1, 1],
        },
        Value::Int(value) => InputFacts {
            class: InputClass::IntegerScalar(value.numeric_dtype()),
            shape: vec![1, 1],
        },
        Value::Bool(_) => InputFacts {
            class: InputClass::LogicalScalar,
            shape: vec![1, 1],
        },
        Value::Complex(_, _) => InputFacts {
            class: InputClass::ComplexScalar,
            shape: vec![1, 1],
        },
        _ => InputFacts {
            class: InputClass::Unsupported,
            shape: vec![],
        },
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

#[cfg(test)]
mod facts_tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage,
        HostNumericDataOwned, HostNumericTensorOwned,
    };
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn identity(value: &Value) -> Digest {
        Digest::sha256(serde_json::to_vec(&input_facts(value)).unwrap())
    }

    #[test]
    fn host_input_signature_reuses_class_and_shape_without_serializing_payloads() {
        let first = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).unwrap(),
        );
        let changed_payload = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![31, 32]), vec![1, 2]).unwrap(),
        );
        let changed_class = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).unwrap(),
        );
        let changed_shape = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap(),
        );
        assert_eq!(identity(&first), identity(&changed_payload));
        assert_ne!(identity(&first), identity(&changed_class));
        assert_ne!(identity(&first), identity(&changed_shape));
        assert_eq!(
            identity(&Value::Int(IntValue::I64(1))),
            identity(&Value::Int(IntValue::I64(i64::MAX)))
        );
        assert_ne!(
            identity(&Value::Int(IntValue::I64(1))),
            identity(&Value::Int(IntValue::U64(1)))
        );
        let encoded = serde_json::to_string(&input_facts(&first)).unwrap();
        assert!(!encoded.contains("31"));
    }

    #[test]
    fn resident_input_signature_preserves_provenance_and_shape() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0, 2.0]),
                shape: vec![1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let automatic = Value::GpuTensor(
            allocation
                .handle
                .clone()
                .with_provenance(GpuHandleProvenance::Automatic)
                .unwrap(),
        );
        let explicit = Value::GpuTensor(
            allocation
                .handle
                .clone()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap(),
        );
        assert_ne!(identity(&automatic), identity(&explicit));
        assert_eq!(input_facts(&automatic).shape, vec![1, 2]);
    }
}
