//! Admitted resident dimension control readback. The source handle stays
//! resident; the transient host payload and callback each retain their own
//! grant until physical retirement.

use crate::builtins::common::gpu_helpers::download_value_preserving_residency_staged_async;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_accelerate_api::{
    checked_numeric_download_host_payload_bytes, handle_is_logical, provider_for_handle,
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderFeasibilityQuery,
    ProviderQueryMetadataFootprint,
};
use runmat_execution::ResourceByteBound;
use runmat_value::{
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType, Value,
};

pub(super) struct DownloadedControl {
    pub(super) value: Value,
    _metadata: HostAllocationReceipt,
    _native: HostAllocationReceipt,
}

fn invalid(detail: &'static str) -> crate::RuntimeError {
    build_runtime_error(detail)
        .with_builtin("reshape")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> crate::RuntimeError {
    build_runtime_error(format!("reshape: dimension download admission: {cause}"))
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn numeric_dtype(element: NumericElementType) -> NumericDType {
    match element {
        NumericElementType::F64 => NumericDType::F64,
        NumericElementType::F32 => NumericDType::F32,
        NumericElementType::I8 => NumericDType::I8,
        NumericElementType::I16 => NumericDType::I16,
        NumericElementType::I32 => NumericDType::I32,
        NumericElementType::I64 => NumericDType::I64,
        NumericElementType::U8 => NumericDType::U8,
        NumericElementType::U16 => NumericDType::U16,
        NumericElementType::U32 => NumericDType::U32,
        NumericElementType::U64 => NumericDType::U64,
    }
}

pub(super) async fn download_control(
    context: &RuntimeContext,
    handle: &GpuTensorHandle,
) -> BuiltinResult<DownloadedControl> {
    if handle.physical().storage() != GpuTensorStorage::Real
        || handle.view().validate(handle.physical()).is_err()
    {
        return Err(invalid(
            "reshape: dimension control must have a live real numeric view",
        ));
    }
    handle.allocation_lease().validate_live().map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            invalid("reshape: dimension control allocation retired"),
            runmat_accelerate_api::ProviderExecutionError::from_source(cause),
        )
    })?;
    let provider = provider_for_handle(handle)
        .ok_or_else(|| invalid("reshape: dimension control has no provider owner"))?;
    if provider.device_id() != handle.device_id() {
        return Err(invalid(
            "reshape: dimension control provider differs from its owner",
        ));
    }
    let rank = handle.shape().len();
    let footprint = ProviderQueryMetadataFootprint {
        operation: "transfer.download",
        input_ranks: &[rank],
        output_ranks: &[rank],
        preparation_shape_ranks: &[rank],
        preparation_representation_slots: 0,
    };
    let query_bytes = footprint
        .requested_bytes()
        .ok_or_else(|| invalid("reshape: dimension download query extent overflow"))?;
    let payload = checked_numeric_download_host_payload_bytes(handle)
        .ok_or_else(|| invalid("reshape: dimension download extent overflow"))?;
    let shape_bytes = rank
        .checked_mul(std::mem::size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("reshape: dimension shape extent overflow"))?;
    let wrapper = if handle_is_logical(handle) {
        // Conversion creates a separate u8 logical buffer while native data
        // remains live. The native payload and shape are in provider workspace.
        u64::try_from(
            ExactHostLogicalTensorAllocation::unmanaged_requested_bytes(handle.shape())
                .map_err(|_| invalid("reshape: logical dimension wrapper overflow"))?,
        )
        .ok()
        .and_then(|bytes| bytes.checked_sub(shape_bytes))
    } else {
        u64::try_from(
            ExactHostTensorAllocation::unmanaged_requested_bytes(
                numeric_dtype(handle.physical().element_type()),
                handle.shape(),
            )
            .map_err(|_| invalid("reshape: numeric dimension wrapper overflow"))?,
        )
        .ok()
        .and_then(|bytes| bytes.checked_sub(payload))
        .and_then(|bytes| bytes.checked_sub(shape_bytes))
    }
    .ok_or_else(|| invalid("reshape: dimension wrapper extent overflow"))?;
    let callback = match provider.download_numeric_completion_host_bound(handle) {
        ResourceByteBound::Known(bytes) => bytes,
        ResourceByteBound::Unavailable(_) => {
            return Err(invalid(
                "reshape: provider has no finite dimension callback bound",
            ));
        }
    };
    let callback_charge = HostAllocationPlan::retained_metadata_bytes(callback)
        .map_err(|_| invalid("reshape: dimension callback extent overflow"))?;
    let retained = query_bytes
        .checked_add(wrapper)
        .and_then(|bytes| bytes.checked_add(callback_charge))
        .ok_or_else(|| invalid("reshape: dimension metadata extent overflow"))?;
    let metadata_plan = HostAllocationPlan::new(0, retained)
        .map_err(|_| invalid("reshape: dimension metadata extent overflow"))?;
    let metadata = HostAllocationReceipt::admit(Some(context), metadata_plan)
        .map_err(|cause| admission_error(context, cause))?
        .ok_or_else(|| invalid("reshape: dimension metadata owner is not managed"))?;
    let callback_claim = metadata
        .claim_retained_metadata(context, callback)
        .map_err(|cause| admission_error(context, cause))?;
    let query = ProviderFeasibilityQuery::exact_numeric_download(handle)
        .ok_or_else(|| invalid("reshape: dimension query could not be prepared"))?;
    if !footprint.matches(&query) || !query.matches_numeric_download(handle) {
        return Err(invalid("reshape: dimension download query changed"));
    }
    if !matches!(
        provider.download_numeric_provider_scratch_bound(handle),
        ResourceByteBound::Known(_)
    ) {
        return Err(invalid(
            "reshape: provider has no finite dimension staging bound",
        ));
    }
    let ResourceByteBound::Known(native_bytes) =
        provider.download_numeric_host_workspace_bound(handle)
    else {
        return Err(invalid(
            "reshape: provider has no finite native download bound",
        ));
    };
    let native_plan = HostAllocationPlan::new(native_bytes, 0)
        .map_err(|_| invalid("reshape: native dimension extent overflow"))?;
    let native = HostAllocationReceipt::admit(Some(context), native_plan)
        .map_err(|cause| admission_error(context, cause))?
        .ok_or_else(|| invalid("reshape: native dimension owner is not managed"))?;
    metadata
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    native
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    let stage = provider
        .prepare_numeric_download_stage(handle, &query, callback_claim)
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                invalid("reshape: provider could not reserve dimension staging"),
                runmat_accelerate_api::ProviderExecutionError::from(source),
            )
        })?;
    if !stage.matches_source(handle) {
        return Err(invalid(
            "reshape: provider staged a different dimension control",
        ));
    }
    let identity = handle.view_identity();
    metadata
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    native
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    let value = download_value_preserving_residency_staged_async(provider, handle, stage).await?;
    metadata
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    native
        .checkpoint(context)
        .map_err(|cause| admission_error(context, cause))?;
    if handle.view_identity() != identity || !query.matches_numeric_download(handle) {
        return Err(invalid(
            "reshape: dimension control changed during readback",
        ));
    }
    Ok(DownloadedControl {
        value,
        _metadata: metadata,
        _native: native,
    })
}
