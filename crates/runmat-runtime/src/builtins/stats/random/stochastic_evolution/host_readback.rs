//! Exact original-input readback for an automatic resident Host alternative.
//! The query and physical bounds are prepared before placement; the transfer
//! executes only after the first ordered t/Z source checkpoints.

use super::source_eligibility::StochasticPreparedSource;
use crate::builtins::common::gpu_helpers::{
    download_value_preserving_residency_admitted_async, download_value_preserving_residency_async,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_accelerate_api::{
    handle_provenance, provider_for_handle, AccelProvider, GpuHandleIdentity, GpuHandleProvenance,
    GpuTensorHandle, ProviderFeasibility, ProviderFeasibilityQuery, ProviderQueryMetadataFootprint,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_execution::ResourceByteBound;
use runmat_value::{ExactHostTensorAllocation, NumericDType, Value};
use std::sync::Arc;

pub(super) struct StochasticHostReadback {
    pub(super) provider: &'static dyn AccelProvider,
    view: GpuHandleIdentity,
    pub(super) query: ProviderFeasibilityQuery,
    host_receipt: Option<HostAllocationReceipt>,
    native_receipt: Option<HostAllocationReceipt>,
    native_admitted: bool,
    callback_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    pub(super) host_workspace: ResourceByteBound,
    pub(super) provider_scratch: ResourceByteBound,
    pub(super) queue_slots: u32,
}

/// Both grants outlive the adopted native buffer, but neither follows a
/// published result: the downloaded input is replaced by the first S value.
pub(super) struct StochasticReadbackRetention {
    _metadata: Option<HostAllocationReceipt>,
    _native_payload: Option<HostAllocationReceipt>,
}

fn invalid_readback(detail: &'static str) -> crate::RuntimeError {
    build_runtime_error(detail)
        .with_identifier("RunMat:StochasticSource:InvalidReadback")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn readback_admission_error(
    context: &RuntimeContext,
    cause: HostAllocationError,
) -> crate::RuntimeError {
    build_runtime_error(format!("automatic source readback admission: {cause}"))
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

impl StochasticHostReadback {
    pub(super) fn prepare(
        context: &RuntimeContext,
        prepared: &StochasticPreparedSource,
    ) -> BuiltinResult<Option<Self>> {
        let Value::GpuTensor(handle) = prepared.state() else {
            return Ok(None);
        };
        if handle_provenance(handle) != Some(GpuHandleProvenance::Automatic) {
            return Ok(None);
        }
        handle.allocation_lease().validate_live().map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                invalid_readback("automatic source allocation retired before selection"),
                runmat_accelerate_api::ProviderExecutionError::from_source(cause),
            )
        })?;
        let Some(provider) = provider_for_handle(handle) else {
            return Err(invalid_readback(
                "automatic source has no live provider owner",
            ));
        };
        if provider.device_id() != handle.device_id()
            || handle.shape() != prepared.pattern().shape()
            || handle.view().validate(handle.physical()).is_err()
        {
            return Err(invalid_readback(
                "automatic source view changed before selection",
            ));
        }
        let rank = handle.shape().len();
        let footprint = ProviderQueryMetadataFootprint {
            operation: "transfer.download",
            input_ranks: &[rank],
            output_ranks: &[rank],
            // The exact constructor materializes one shape before cloning it
            // into the input and output representations. Feasibility also
            // constructs one transient exact-match query beside this one.
            preparation_shape_ranks: &[rank],
            preparation_representation_slots: 0,
        };
        let bytes = footprint
            .requested_bytes()
            .ok_or_else(|| invalid_readback("download query extent overflow"))?;
        let dtype = match prepared.pattern().state_class() {
            runmat_types::NumericClass::Single => NumericDType::F32,
            runmat_types::NumericClass::Double => NumericDType::F64,
            _ => return Ok(None),
        };
        let total_output = ExactHostTensorAllocation::requested_bytes(dtype, handle.shape())
            .ok()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| invalid_readback("download Tensor metadata extent overflow"))?;
        let payload = runmat_accelerate_api::checked_numeric_download_host_payload_bytes(handle)
            .ok_or_else(|| invalid_readback("download payload extent overflow"))?;
        let native_shape = handle
            .shape()
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| invalid_readback("download shape extent overflow"))?;
        let wrapper_metadata = total_output
            .checked_sub(payload)
            .and_then(|bytes| bytes.checked_sub(native_shape))
            .ok_or_else(|| invalid_readback("download Tensor metadata extent overflow"))?;
        let retained = bytes
            .checked_add(wrapper_metadata)
            .ok_or_else(|| invalid_readback("download metadata extent overflow"))?;
        let host_workspace = provider.download_numeric_host_workspace_bound(handle);
        if !matches!(host_workspace, ResourceByteBound::Known(_)) {
            return Ok(None);
        }
        let callback_bound = provider.download_numeric_completion_host_bound(handle);
        let finite = context.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        let callback_bytes = match callback_bound {
            ResourceByteBound::Known(bytes) => Some(bytes),
            ResourceByteBound::Unavailable(_) if finite => return Ok(None),
            ResourceByteBound::Unavailable(_) => None,
        };
        let callback_claim_bytes = callback_bytes
            .map(HostAllocationPlan::retained_metadata_bytes)
            .transpose()
            .map_err(|_| invalid_readback("download callback extent overflow"))?
            .unwrap_or(0);
        let retained = retained
            .checked_add(callback_claim_bytes)
            .ok_or_else(|| invalid_readback("download metadata extent overflow"))?;
        let plan = HostAllocationPlan::new(bytes, retained)
            .map_err(|_| invalid_readback("download query extent overflow"))?;
        let host_receipt = match HostAllocationReceipt::admit(Some(context), plan) {
            Ok(receipt) => receipt,
            Err(cause) if cause.is_capacity_denial() => return Ok(None),
            Err(cause) => return Err(readback_admission_error(context, cause)),
        };
        let callback_claim = match (host_receipt.as_ref(), callback_bytes) {
            (Some(receipt), Some(bytes)) => match receipt.claim_retained_metadata(context, bytes) {
                Ok(claim) => Some(claim),
                Err(cause) if cause.is_capacity_denial() => return Ok(None),
                Err(cause) => return Err(readback_admission_error(context, cause)),
            },
            _ => None,
        };
        let Some(query) = ProviderFeasibilityQuery::exact_numeric_download(handle) else {
            return Ok(None);
        };
        if !footprint.matches(&query) {
            return Err(invalid_readback(
                "download query differs from admitted metadata",
            ));
        }
        if !matches!(
            provider.query_numeric_download_feasibility(handle, &query),
            ProviderFeasibility::Supported { .. }
        ) {
            return Ok(None);
        }
        Ok(Some(Self {
            provider,
            view: handle.view_identity(),
            query,
            host_receipt,
            native_receipt: None,
            native_admitted: false,
            callback_claim,
            host_workspace,
            provider_scratch: provider.download_numeric_provider_scratch_bound(handle),
            queue_slots: provider.download_numeric_queue_slots(handle),
        }))
    }

    /// The optional Host route is selected before this grant. Reserving the
    /// native payload during speculative preparation could evict a legal
    /// Provider route. Capacity refusal is still before the first source
    /// store or RNG draw, so the original For remains available.
    pub(super) fn admit_selected_native(
        &mut self,
        context: &RuntimeContext,
    ) -> BuiltinResult<bool> {
        let ResourceByteBound::Known(native_bytes) = self.host_workspace else {
            return Ok(false);
        };
        let plan = HostAllocationPlan::new(native_bytes, 0)
            .map_err(|_| invalid_readback("native download extent overflow"))?;
        match HostAllocationReceipt::admit(Some(context), plan) {
            Ok(receipt) => {
                self.native_receipt = receipt;
                self.native_admitted = true;
                Ok(true)
            }
            Err(cause) if cause.is_capacity_denial() => Ok(false),
            Err(cause) => Err(readback_admission_error(context, cause)),
        }
    }

    pub(super) async fn download(
        self,
        context: &RuntimeContext,
        handle: &GpuTensorHandle,
    ) -> BuiltinResult<(Value, StochasticReadbackRetention)> {
        if !self.native_admitted {
            return Err(invalid_readback(
                "selected native readback was not admitted",
            ));
        }
        if let Some(receipt) = self.host_receipt.as_ref() {
            receipt.checkpoint(context).map_err(|cause| {
                build_runtime_error(format!("automatic source readback admission: {cause}"))
                    .with_identifier(format!(
                        "{}:{}",
                        context.error_namespace(),
                        cause.runtime_identifier_suffix()
                    ))
                    .with_source(cause)
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build()
            })?;
        }
        if let Some(receipt) = self.native_receipt.as_ref() {
            receipt.checkpoint(context).map_err(|cause| {
                build_runtime_error(format!(
                    "automatic source native readback admission: {cause}"
                ))
                .with_identifier(format!(
                    "{}:{}",
                    context.error_namespace(),
                    cause.runtime_identifier_suffix()
                ))
                .with_source(cause)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build()
            })?;
        }
        handle.allocation_lease().validate_live().map_err(|cause| {
            crate::runtime_error::with_provider_execution_source(
                invalid_readback("selected automatic source allocation retired"),
                runmat_accelerate_api::ProviderExecutionError::from_source(cause),
            )
        })?;
        if handle.view_identity() != self.view
            || provider_for_handle(handle).is_none_or(|provider| {
                !runmat_accelerate_api::same_provider_instance(provider, self.provider)
            })
            || handle_provenance(handle) != Some(GpuHandleProvenance::Automatic)
            || !self.query.matches_numeric_download(handle)
        {
            return Err(invalid_readback(
                "selected automatic readback source changed",
            ));
        }
        let value = match self.callback_claim {
            Some(claim) => {
                download_value_preserving_residency_admitted_async(self.provider, handle, claim)
                    .await?
            }
            None => download_value_preserving_residency_async(self.provider, handle).await?,
        };
        if let Some(receipt) = self.host_receipt.as_ref() {
            receipt.checkpoint(context).map_err(|cause| {
                build_runtime_error(format!("automatic source readback admission: {cause}"))
                    .with_identifier(format!(
                        "{}:{}",
                        context.error_namespace(),
                        cause.runtime_identifier_suffix()
                    ))
                    .with_source(cause)
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build()
            })?;
        }
        if let Some(receipt) = self.native_receipt.as_ref() {
            receipt.checkpoint(context).map_err(|cause| {
                build_runtime_error(format!(
                    "automatic source native readback admission: {cause}"
                ))
                .with_identifier(format!(
                    "{}:{}",
                    context.error_namespace(),
                    cause.runtime_identifier_suffix()
                ))
                .with_source(cause)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build()
            })?;
        }
        Ok((
            value,
            StochasticReadbackRetention {
                _metadata: self.host_receipt,
                _native_payload: self.native_receipt,
            },
        ))
    }
}
