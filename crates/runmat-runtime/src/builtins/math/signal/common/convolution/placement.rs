//! Whole-call placement for the two live-handle convolution hooks. Public
//! parsing and semantic admission remain with the identity owners.

#[cfg(test)]
#[path = "placement/tests.rs"]
pub(crate) mod tests;

use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationSupport, ProviderOperationReceipt,
    ProviderOperationRecorder,
};
use runmat_accelerate_api::{
    same_provider_instance, GpuTensorHandle, ProviderConvolutionForm,
    ProviderConvolutionInputBackings, ProviderConvolutionWorkload, ProviderElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRejection, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderTransferOutputBound, ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, ExecutionCandidateKind, PlacementResourceSnapshot,
    PlacementSignature, ResourceBoundUnavailable, ResourceByteBound, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkPlacementRequest,
};
use runmat_types::ConvolutionMode;
use runmat_value::Value;

use crate::context::{
    host_presentation::HostPresentationReceipt, RuntimeContext, RuntimeWholeWorkSelection,
};
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

#[derive(Debug)]
pub(crate) enum LateUploadAdmissionError {
    OutputCount {
        expected: usize,
        actual: usize,
    },
    UnsupportedOutput {
        index: usize,
    },
    ShapeOverflow {
        index: usize,
    },
    BoundMismatch {
        index: usize,
        actual: ProviderRepresentation,
        bound: ProviderTransferOutputBound,
    },
    ProviderRejected {
        rejection: ProviderRejection,
    },
}

impl std::fmt::Display for LateUploadAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputCount { expected, actual } => {
                write!(f, "expected {expected} outputs, got {actual}")
            }
            Self::UnsupportedOutput { index } => write!(
                f,
                "output {} has no admitted dense real transfer representation",
                index + 1
            ),
            Self::ShapeOverflow { index } => write!(
                f,
                "output {} shape exceeds provider representation limits",
                index + 1
            ),
            Self::BoundMismatch {
                index,
                actual,
                bound,
            } => write!(
                f,
                "output {} representation {actual:?} exceeds admitted bound {bound:?}",
                index + 1
            ),
            Self::ProviderRejected { rejection } => write!(
                f,
                "provider rejected exact upload: {:?} ({:?})",
                rejection.code, rejection.detail
            ),
        }
    }
}

impl std::error::Error for LateUploadAdmissionError {}

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
    ) -> Result<(), LateUploadAdmissionError> {
        let Some(bounds) = &self.expected_bounds else {
            return Ok(());
        };
        let validate = || {
            if values.len() != bounds.len() {
                return Err(LateUploadAdmissionError::OutputCount {
                    expected: bounds.len(),
                    actual: values.len(),
                });
            }
            let mut actual = Vec::with_capacity(values.len());
            for (index, (value, bound)) in values.iter().zip(bounds).enumerate() {
                let (element_type, shape) = match value {
                    Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
                        let shape = tensor
                            .shape()
                            .iter()
                            .copied()
                            .map(u64::try_from)
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|_| LateUploadAdmissionError::ShapeOverflow { index })?;
                        let element_type =
                            if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
                                ProviderElementType::F32
                            } else {
                                ProviderElementType::F64
                            };
                        (element_type, shape)
                    }
                    Value::Num(_) => (ProviderElementType::F64, vec![1, 1]),
                    _ => return Err(LateUploadAdmissionError::UnsupportedOutput { index }),
                };
                let representation = ProviderRepresentation {
                    element_type,
                    storage: ProviderStorage::DenseReal,
                    layout: ProviderLayout::ColumnMajorContiguous,
                    shape,
                    residency: ProviderResidency::Device,
                };
                if !bound.admits(&representation) {
                    return Err(LateUploadAdmissionError::BoundMismatch {
                        index,
                        actual: representation,
                        bound: bound.clone(),
                    });
                }
                actual.push(representation);
            }
            let query = ProviderFeasibilityQuery {
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
                outputs: actual,
                workload: ProviderWorkload {
                    requested_outputs: u32::try_from(values.len()).ok(),
                    ..ProviderWorkload::default()
                },
            };
            match provider.query_feasibility(&query) {
                ProviderFeasibility::Supported { .. } => Ok(()),
                ProviderFeasibility::Rejected { rejection } => {
                    Err(LateUploadAdmissionError::ProviderRejected { rejection })
                }
            }
        };
        let result = validate();
        if result.is_err() {
            if let Some(recorder) = self.recorder() {
                recorder.fail(ProviderObservationFailure::SharedRuntimeFallback);
            }
        }
        result
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
        if let Some(receipt) = self.receipt.take() {
            let published = if outcome
                .as_ref()
                .err()
                .is_some_and(|error| error.identifier() == Some("RunMat:Cancelled"))
            {
                Err(ProviderObservationFailure::Cancelled)
            } else if outcome.is_err() {
                Err(ProviderObservationFailure::ProviderFailed)
            } else {
                Ok(true)
            };
            HostPresentationReceipt::new(self.runtime, self.selected, receipt).finish(published);
        }
    }

    pub(crate) fn finish_value(mut self, outcome: &BuiltinResult<runmat_value::Value>) {
        if let Some(receipt) = self.receipt.take() {
            let published = match outcome {
                Err(error) if error.identifier() == Some("RunMat:Cancelled") => {
                    Err(ProviderObservationFailure::Cancelled)
                }
                Err(_) => Err(ProviderObservationFailure::ProviderFailed),
                Ok(value) => Ok(self.published_matches(value, 0)),
            };
            HostPresentationReceipt::new(self.runtime, self.selected, receipt).finish(published);
        }
    }

    /// Multi-output host routes use the same one-shot receipt and feedback
    /// owner; each requested output must actually be restored to the owner.
    pub(crate) fn finish_outputs(mut self, values: &[&Value], expected: usize) {
        if let Some(receipt) = self.receipt.take() {
            let published = values.len() == expected
                && values
                    .iter()
                    .enumerate()
                    .all(|(index, value)| self.published_matches(value, index));
            HostPresentationReceipt::new(self.runtime, self.selected, receipt)
                .finish(Ok(published));
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
}

mod host;
mod pair;

pub(crate) use host::{
    host_output_representations, select_bounded_host_variant, select_host_variant,
};
pub(crate) use pair::select_pair;
