//! Whole-call placement for the two live-handle convolution hooks. Public
//! parsing and semantic admission remain with the identity owners.

#[cfg(test)]
#[path = "placement/tests.rs"]
pub(crate) mod tests;

use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderObservationSupport,
    ProviderOperationReceipt, ProviderOperationRecorder,
};
use runmat_accelerate_api::{
    same_provider_instance, GpuTensorHandle, ProviderConvolutionForm,
    ProviderConvolutionInputBackings, ProviderConvolutionWorkload, ProviderElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderTransferOutputBound, ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementFeedback,
    PlacementResourceSnapshot, PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
    WholeWorkCandidate, WholeWorkConstraint, WholeWorkPlacementRequest,
};
use runmat_time::duration_ns_saturating;
use runmat_types::ConvolutionMode;
use runmat_value::Value;

use crate::context::{RuntimeContext, RuntimePlacementFeedbackSource, RuntimeWholeWorkSelection};
use crate::BuiltinResult;

/// The receipt is tied to the actual borrowed pair, not a serialized shape.
/// Dropping a pending call releases its exploration probe without feedback.
pub(crate) struct SelectedConvolution {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    provider_route: bool,
    backing: Option<ProviderConvolutionInputBackings>,
    receipt: Option<ProviderOperationReceipt>,
    expected_outputs: Vec<ProviderRepresentation>,
    expected_bounds: Option<Vec<ProviderTransferOutputBound>>,
    expected_device: u32,
}

impl SelectedConvolution {
    pub(crate) fn provider_route(&self) -> bool {
        self.provider_route
    }
    pub(crate) fn recorder(&self) -> Option<ProviderOperationRecorder> {
        self.receipt
            .as_ref()
            .map(ProviderOperationReceipt::recorder)
    }

    /// Late numerical dimensions can narrow a previously admitted bound, but
    /// cannot enlarge it or replace the whole-call placement decision.
    pub(crate) fn validate_actual_host_outputs(
        &self,
        provider: &dyn runmat_accelerate_api::AccelProvider,
        values: &[&Value],
    ) -> bool {
        let Some(bounds) = &self.expected_bounds else {
            return true;
        };
        let actual = values
            .iter()
            .map(|value| {
                let (element_type, shape) = match value {
                    Value::Tensor(tensor) if tensor.integer_storage().is_none() => (
                        if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
                            ProviderElementType::F32
                        } else {
                            ProviderElementType::F64
                        },
                        tensor
                            .shape
                            .iter()
                            .copied()
                            .map(u64::try_from)
                            .collect::<Result<Vec<_>, _>>()
                            .ok()?,
                    ),
                    Value::Num(_) => (ProviderElementType::F64, vec![1, 1]),
                    _ => return None,
                };
                Some(ProviderRepresentation {
                    element_type,
                    storage: ProviderStorage::DenseReal,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    shape,
                    residency: ProviderResidency::Device,
                })
            })
            .collect::<Option<Vec<_>>>();
        let valid = actual.as_ref().is_some_and(|actual| {
            actual.len() == bounds.len()
                && actual
                    .iter()
                    .zip(bounds)
                    .all(|(representation, bound)| bound.admits(representation))
                && provider
                    .query_feasibility(&ProviderFeasibilityQuery {
                        operation: ProviderOperationIdentity::new("transfer.upload"),
                        family: ProviderOperationFamily::Upload,
                        inputs: actual
                            .iter()
                            .cloned()
                            .map(|mut input| {
                                input.residency = ProviderResidency::Host;
                                input
                            })
                            .collect(),
                        outputs: actual.clone(),
                        workload: ProviderWorkload {
                            requested_outputs: u32::try_from(actual.len()).ok(),
                            ..ProviderWorkload::default()
                        },
                    })
                    .is_supported()
        });
        if !valid {
            if let Some(recorder) = self.recorder() {
                recorder.fail(ProviderObservationFailure::SharedRuntimeFallback);
            }
        }
        valid
    }

    pub(crate) fn revalidate(&self, handles: [&GpuTensorHandle; 2]) -> BuiltinResult<bool> {
        let Some(backing) = self.backing else {
            return Ok(false);
        };
        backing.matches_live_handles(handles).map_err(|cause| {
            crate::build_runtime_error("convolution input lease expired before dispatch")
                .with_identifier("RunMat:Placement:ExpiredConvolutionInput")
                .with_source(cause)
                .build()
        })
    }

    pub(crate) fn finish_hook(mut self, outcome: &BuiltinResult<Option<runmat_value::Value>>) {
        if let Some(mut receipt) = self.receipt.take() {
            if self.runtime.is_cancelled()
                || outcome
                    .as_ref()
                    .err()
                    .is_some_and(|error| error.identifier() == Some("RunMat:Cancelled"))
            {
                receipt.fail(ProviderObservationFailure::Cancelled);
            } else if outcome.is_err() {
                receipt.fail(ProviderObservationFailure::ProviderFailed);
            } else if receipt.publish().is_err() {
                receipt.fail(ProviderObservationFailure::PublicationFailed);
            }
            self.deliver(receipt);
        }
    }

    pub(crate) fn finish_value(mut self, outcome: &BuiltinResult<runmat_value::Value>) {
        if let Some(mut receipt) = self.receipt.take() {
            if self.runtime.is_cancelled()
                || outcome
                    .as_ref()
                    .err()
                    .is_some_and(|error| error.identifier() == Some("RunMat:Cancelled"))
            {
                receipt.fail(ProviderObservationFailure::Cancelled);
            } else if outcome.is_err() {
                receipt.fail(ProviderObservationFailure::ProviderFailed);
            } else if !outcome
                .as_ref()
                .is_ok_and(|value| self.published_matches(value, 0))
            {
                receipt.fail(ProviderObservationFailure::SharedRuntimeFallback);
            } else if receipt.publish().is_err() {
                receipt.fail(ProviderObservationFailure::PublicationFailed);
            }
            self.deliver(receipt);
        }
    }

    /// Multi-output host routes use the same one-shot receipt and feedback
    /// owner; each requested output must actually be restored to the owner.
    pub(crate) fn finish_outputs(mut self, values: &[&Value], expected: usize) {
        if let Some(mut receipt) = self.receipt.take() {
            if self.runtime.is_cancelled() {
                receipt.fail(ProviderObservationFailure::Cancelled);
            } else if values.len() != expected
                || !values
                    .iter()
                    .enumerate()
                    .all(|(index, value)| self.published_matches(value, index))
            {
                receipt.fail(ProviderObservationFailure::SharedRuntimeFallback);
            } else if receipt.publish().is_err() {
                receipt.fail(ProviderObservationFailure::PublicationFailed);
            }
            self.deliver(receipt);
        }
    }

    fn published_matches(&self, value: &Value, index: usize) -> bool {
        let Value::GpuTensor(handle) = value else {
            return false;
        };
        let Some(shape) = handle
            .shape()
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()
        else {
            return false;
        };
        let expected = if let Some(exact) = self.expected_outputs.get(index) {
            exact.clone()
        } else if let Some(bound) = self
            .expected_bounds
            .as_ref()
            .and_then(|bounds| bounds.get(index))
        {
            let actual = ProviderRepresentation {
                element_type: bound.element_type,
                storage: bound.storage,
                layout: bound.layout,
                shape,
                residency: bound.residency,
            };
            if !bound.admits(&actual) {
                return false;
            }
            actual
        } else {
            return false;
        };
        handle.device_id() == self.expected_device
            && expected
                .shape
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()
                .as_deref()
                == Some(handle.shape())
            && runmat_accelerate_api::provider_for_handle(handle).is_some()
            && match expected.element_type {
                ProviderElementType::F32 => {
                    runmat_accelerate_api::handle_precision(handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F32)
                }
                ProviderElementType::F64 => {
                    runmat_accelerate_api::handle_precision(handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F64)
                }
                _ => false,
            }
            && runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
    }

    fn deliver(&self, receipt: ProviderOperationReceipt) {
        let mut deferred = DeferredConvolutionFeedback {
            receipt,
            signature: self.selected.decision.signature.clone(),
            selected: self.selected.decision.selected.clone(),
            cancellation: self.runtime.cancellation(),
            terminal_without_feedback: false,
        };
        if let Some(service) = self.runtime.service_ports().placement() {
            if let Some(ready) = deferred.take_ready() {
                let _ = service.observe(ready);
            } else if !deferred.terminal_without_feedback()
                && service.defer_feedback(Box::new(deferred)).is_err()
            {
                tracing::debug!(target: "runmat_placement", "convolution completion receipt could not be retained");
            }
        }
    }
}

struct DeferredConvolutionFeedback {
    receipt: ProviderOperationReceipt,
    signature: PlacementSignature,
    selected: runmat_execution::SelectedExecutionCandidate,
    cancellation: runmat_execution::cancellation::CancellationToken,
    terminal_without_feedback: bool,
}

impl RuntimePlacementFeedbackSource for DeferredConvolutionFeedback {
    fn signature(&self) -> &PlacementSignature {
        &self.signature
    }

    fn take_ready(&mut self) -> Option<PlacementFeedback> {
        if self.cancellation.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        }
        let observation = self.receipt.take_completed()?;
        let (elapsed, succeeded) = match observation.status {
            ProviderObservationStatus::Completed { total_elapsed, .. } => {
                (duration_ns_saturating(total_elapsed), true)
            }
            ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed) => {
                (0, false)
            }
            ProviderObservationStatus::Failed(_) => {
                self.terminal_without_feedback = true;
                return None;
            }
            ProviderObservationStatus::Pending => return None,
        };
        Some(PlacementFeedback {
            signature: self.signature.clone(),
            candidate: self.selected.candidate.clone(),
            kind: self.selected.kind,
            total_elapsed_ns: elapsed,
            succeeded,
        })
    }

    fn terminal_without_feedback(&self) -> bool {
        self.terminal_without_feedback
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
