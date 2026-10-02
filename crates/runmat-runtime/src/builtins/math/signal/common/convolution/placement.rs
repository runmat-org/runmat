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

mod host;
mod pair;

pub(crate) use host::{
    host_output_representations, select_bounded_host_variant, select_host_variant,
};
pub(crate) use pair::select_pair;
