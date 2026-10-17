//! One preselected host-compute, native-upload route for GPU child generators.
//! The result class can differ from a Like prototype (randperm is Double),
//! but the provider and output intent can never silently change.

use crate::builtins::common::gpu_helpers;
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
    HostNumericWorkspace,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    class_preserving_upload_query, upload_numeric_protecting_admitted, AccelProvider,
    GpuTensorHandle, HostNumericTensorView, NumericElementType, ProviderElementType,
    ProviderFeasibility, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use runmat_execution::{
    single_candidate_resources_admitted, CandidateExecutionLocation, CandidateOutputPresentation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand,
    CandidateResourceFacts, ExecutionCandidateKind, PlacementResourceSnapshot, ResourceByteBound,
};
use runmat_value::NumericDType;

fn error(label: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
}

/// A committed child-stream range is a semantic effect even if the provider
/// has not submitted device work. Preserve the original language error but
/// prohibit a caller from replaying the operation on another route.
pub(crate) fn after_stream_commit(
    label: &'static str,
    source: crate::RuntimeError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        error(label, source.message.clone()),
        runmat_accelerate_api::ProviderExecutionError::from_source(source).after_effect(),
    )
}

fn provider_element(class: NumericElementType) -> ProviderElementType {
    match class {
        NumericElementType::F64 => ProviderElementType::F64,
        NumericElementType::F32 => ProviderElementType::F32,
        NumericElementType::I8 => ProviderElementType::I8,
        NumericElementType::I16 => ProviderElementType::I16,
        NumericElementType::I32 => ProviderElementType::I32,
        NumericElementType::I64 => ProviderElementType::I64,
        NumericElementType::U8 => ProviderElementType::U8,
        NumericElementType::U16 => ProviderElementType::U16,
        NumericElementType::U32 => ProviderElementType::U32,
        NumericElementType::U64 => ProviderElementType::U64,
    }
}

pub(crate) struct GpuChildHostUpload<'a> {
    label: &'static str,
    provider: &'static dyn AccelProvider,
    source: &'a GpuTensorHandle,
    shape: &'a [usize],
    class: NumericElementType,
    physical_bytes: u64,
    runtime: Option<crate::context::RuntimeContext>,
    // Rust drops fields in declaration order. The permit releases before the
    // managed headroom owner, including on every early return.
    permit: Option<Box<dyn ProviderHostMetadataAdmission>>,
    receipt: Option<HostAllocationReceipt>,
}

impl<'a> GpuChildHostUpload<'a> {
    /// `compute_scratch_bytes` excludes the single native result payload.
    /// All resource-denying checks occur before a stream range is reserved.
    pub(crate) fn prepare(
        label: &'static str,
        source: &'a GpuTensorHandle,
        shape: &'a [usize],
        class: NumericElementType,
        compute_scratch_bytes: u64,
    ) -> BuiltinResult<Self> {
        let provider = gpu_helpers::exact_provider_for_handle(source).ok_or_else(|| {
            error(
                label,
                format!("{label}: no provider owns the GPU prototype"),
            )
        })?;
        source.allocation_lease().validate_live().map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                error(
                    label,
                    format!("{label}: GPU prototype lease is not live: {cause}"),
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                    .with_unproven_effects(),
            )
        })?;
        let len = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1_usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| error(label, format!("{label}: output extent overflow")))?
        };
        let native_bytes = u64::try_from(len)
            .ok()
            .and_then(|len| len.checked_mul(u64::try_from(class.element_size()).ok()?))
            .ok_or_else(|| error(label, format!("{label}: native output extent overflow")))?;
        let output = ProviderRepresentation {
            element_type: provider_element(class),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape
                .iter()
                .map(|&n| u64::try_from(n))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| error(label, format!("{label}: output dimension overflow")))?,
            residency: ProviderResidency::Device,
        };
        let query = class_preserving_upload_query(vec![output])
            .ok_or_else(|| error(label, format!("{label}: no native upload representation")))?;
        let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
            return Err(error(
                label,
                format!("{label}: provider cannot upload this native class"),
            ));
        };
        let physical = provider.upload_physical_evidence(&query).ok_or_else(|| {
            error(
                label,
                format!("{label}: provider has no physical upload evidence"),
            )
        })?;
        if estimate.output_bytes != Some(physical.output_requested_bytes) {
            return Err(error(
                label,
                format!("{label}: physical output estimate disagrees"),
            ));
        }
        let runtime = crate::context::legacy::active();
        let registration = provider_registration_host_quote(provider, shape.len(), 1, 1)
            .map_err(|source| error(label, format!("{label}: metadata quote: {source}")))?;
        let parallel = runtime
            .as_ref()
            .and_then(|runtime| runtime.service_ports().parallel())
            .map(|service| service.placement_resources())
            .unwrap_or_default();
        if parallel.memory_available_bytes.is_some() && registration.is_none() {
            return Err(error(
                label,
                format!("{label}: provider lacks finite host metadata evidence"),
            ));
        }
        let metadata_bytes = match registration {
            Some(quote) => quote
                .preparation_bytes
                .checked_add(quote.retained_bytes)
                .ok_or_else(|| error(label, format!("{label}: metadata extent overflow")))?,
            None => 0,
        };
        let presentation = CandidateOutputPresentation::ProviderUpload {
            device_id: provider.device_id(),
            transfer_bytes: Some(physical.output_requested_bytes),
            host_result_bytes: Some(ResourceByteBound::Known(native_bytes)),
            host_scratch_bytes: physical.host_workspace_bytes,
            provider_scratch_bytes: physical.provider_scratch_bytes,
            queue_slots: physical.completion_queue_slots,
            preparation: CandidatePreparationState::Ready,
        };
        single_candidate_resources_admitted(
            CandidateResourceFacts {
                location: CandidateExecutionLocation::Host,
                output_residency: CandidateOutputResidency::Provider {
                    device_id: provider.device_id(),
                },
                presentation: &presentation,
                kind: ExecutionCandidateKind::SharedRuntime,
                demand: CandidateResourceDemand {
                    cpu_millicores: 1_000,
                    retained_bytes: ResourceByteBound::Known(physical.output_requested_bytes),
                    scratch_bytes: ResourceByteBound::Known(
                        compute_scratch_bytes
                            .checked_add(metadata_bytes)
                            .ok_or_else(|| error(label, format!("{label}: host peak overflow")))?,
                    ),
                    provider_host_workspace: None,
                    provider_input_stages: None,
                    queue_slots: 0,
                    peak_live: None,
                },
                transactional_results: true,
            },
            &PlacementResourceSnapshot {
                cpu_millicores_available: parallel.cpu_millicores_available,
                memory_available_bytes: parallel.memory_available_bytes,
                cancellation_requested: runtime
                    .as_ref()
                    .is_some_and(|runtime| runtime.is_cancelled()),
                providers: vec![provider.placement_resources()],
                epoch: parallel.epoch,
            },
            false,
        )
        .map_err(|source| {
            error(
                label,
                format!("{label}: host-mediated route not admitted: {source}"),
            )
        })?;
        let (scratch, retained) = match registration {
            Some(quote) => (
                compute_scratch_bytes
                    .checked_add(native_bytes)
                    .and_then(|bytes| bytes.checked_add(quote.preparation_bytes))
                    .ok_or_else(|| error(label, format!("{label}: host extent overflow")))?,
                quote.retained_bytes,
            ),
            None => (
                compute_scratch_bytes
                    .checked_add(native_bytes)
                    .ok_or_else(|| error(label, format!("{label}: host extent overflow")))?,
                0,
            ),
        };
        let receipt = HostAllocationPlan::new(scratch, retained)
            .and_then(|plan| HostAllocationReceipt::admit(runtime.as_ref(), plan))
            .map_err(|source| error(label, format!("{label}: host allocation: {source}")))?;
        let permit = if let (Some(runtime), Some(receipt)) = (runtime.as_ref(), receipt.as_ref()) {
            let retained = receipt
                .retained_view_permit(runtime)
                .map_err(|source| error(label, format!("{label}: metadata permit: {source}")))?;
            Some(
                provider
                    .prepare_output_host_metadata(Box::new(retained), 1)
                    .map_err(|source| {
                        error(label, format!("{label}: provider metadata: {source}"))
                    })?,
            )
        } else {
            None
        };
        Ok(Self {
            label,
            provider,
            source,
            shape,
            class,
            physical_bytes: physical.output_requested_bytes,
            runtime,
            permit,
            receipt,
        })
    }

    pub(crate) fn runtime(&self) -> Option<&crate::context::RuntimeContext> {
        self.runtime.as_ref()
    }

    pub(crate) fn sample_workspace(
        &self,
        dtype: NumericDType,
        len: usize,
    ) -> BuiltinResult<Option<HostNumericWorkspace>> {
        let (Some(runtime), Some(receipt)) = (self.runtime.as_ref(), self.receipt.as_ref()) else {
            return Ok(None);
        };
        receipt
            .temporary_numeric(runtime, dtype, len)
            .map(Some)
            .map_err(|source| {
                error(
                    self.label,
                    format!("{}: sample workspace: {source}", self.label),
                )
            })
    }

    pub(crate) fn checkpoint(&self) -> BuiltinResult<()> {
        if let (Some(runtime), Some(receipt)) = (self.runtime.as_ref(), self.receipt.as_ref()) {
            receipt
                .checkpoint(runtime)
                .map_err(|source| error(self.label, format!("{}: {source}", self.label)))?;
        }
        Ok(())
    }

    /// Caller declares this after the plan so its native sample buffer drops
    /// first. This function does not consume/refund the plan before that drop.
    pub(crate) fn publish(
        &mut self,
        view: &HostNumericTensorView<'_>,
    ) -> BuiltinResult<GpuTensorHandle> {
        self.checkpoint()?;
        self.source
            .allocation_lease()
            .validate_live()
            .map_err(|cause| {
                crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!(
                            "{}: GPU prototype retired before upload: {cause}",
                            self.label
                        ),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                        .with_unproven_effects(),
                )
            })?;
        if view.shape != self.shape
            || view.data.element_type() != self.class
            || view.storage != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return Err(error(
                self.label,
                format!("{}: native result contract mismatch", self.label),
            ));
        }
        let mut output = upload_numeric_protecting_admitted(
            self.provider,
            view,
            &[self.source],
            self.permit.as_deref(),
        )
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                error(
                    self.label,
                    format!("{}: native upload failed: {source}", self.label),
                ),
                source.into_execution_error(),
            )
        })?;
        if !gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
            runmat_accelerate_api::same_provider_instance(owner, self.provider)
        }) || runmat_accelerate_api::handle_class_identity(&output)
            != Some(self.class.class_identity())
            || output.physical().requested_bytes() != self.physical_bytes
        {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                error(
                    self.label,
                    format!(
                        "{}: uploaded result owner/class/physical bound mismatch",
                        self.label
                    ),
                ),
            ));
        }
        if let Err(cause) = gpu_helpers::propagate_output_provenance(&mut output, [self.source]) {
            return Err(gpu_helpers::reject_gpu_output(output, cause));
        }
        if let Err(cause) = self.source.allocation_lease().validate_live() {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                crate::runtime_error::with_provider_execution_source(
                    error(
                        self.label,
                        format!(
                            "{}: GPU prototype retired during upload: {cause}",
                            self.label
                        ),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                        .with_unproven_effects(),
                ),
            ));
        }
        if let Err(cause) = self.checkpoint() {
            return Err(gpu_helpers::reject_gpu_output(output, cause));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod stream_effect_tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn committed_child_range_makes_later_failure_terminal() {
        let error = after_stream_commit("randi", error("randi", "sample allocation failed"));
        let source = error.source().expect("typed stream effect source");
        let typed = source
            .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            .expect("provider execution disposition");
        assert!(typed.is_after_effect());
        assert!(!typed.is_unsupported());
    }
}
