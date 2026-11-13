//! Checked native download geometry before any provider effect.
use super::{invalid, ExactNativeDownloadQuote};
use crate::context::host_allocation::HostAllocationPlan;
use crate::RuntimeError;
use runmat_accelerate_api::{
    checked_numeric_download_host_payload_bytes, handle_is_logical, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, NumericElementType, ProviderQueryMetadataFootprint,
};
use runmat_execution::ResourceByteBound;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType,
};
use std::mem::size_of;

pub fn quote_exact_native_download(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<ExactNativeDownloadQuote, RuntimeError> {
    if provider.device_id() != handle.device_id() {
        return Err(invalid(
            "selected finite readback requires the source provider",
        ));
    }
    quote_geometry(
        handle,
        provider.download_numeric_host_workspace_bound(handle),
        provider.download_numeric_provider_scratch_bound(handle),
        provider.download_numeric_completion_host_bound(handle),
    )
}

pub fn quote_exact_native_readback(
    readback: &runmat_accelerate_api::NumericReadback<'_>,
) -> Result<ExactNativeDownloadQuote, RuntimeError> {
    let failure = |source| {
        crate::runtime_error::with_provider_execution_source(
            invalid("numeric readback authorization or bound failed"),
            runmat_accelerate_api::ProviderExecutionError::from(source),
        )
    };
    quote_geometry(
        readback.source(),
        readback
            .download_numeric_host_workspace_bound()
            .map_err(failure)?,
        readback
            .download_numeric_provider_scratch_bound()
            .map_err(failure)?,
        readback
            .download_numeric_completion_host_bound()
            .map_err(failure)?,
    )
}

fn quote_geometry(
    handle: &GpuTensorHandle,
    workspace_bound: ResourceByteBound,
    scratch_bound: ResourceByteBound,
    callback_bound: ResourceByteBound,
) -> Result<ExactNativeDownloadQuote, RuntimeError> {
    let storage = handle
        .descriptor()
        .storage
        .ok_or_else(|| invalid("selected source has no native storage layout"))?;
    let logical = handle_is_logical(handle);
    let dtype = match handle.physical().element_type() {
        NumericElementType::F32 => NumericDType::F32,
        NumericElementType::F64 => NumericDType::F64,
        NumericElementType::I8 => NumericDType::I8,
        NumericElementType::I16 => NumericDType::I16,
        NumericElementType::I32 => NumericDType::I32,
        NumericElementType::I64 => NumericDType::I64,
        NumericElementType::U8 => NumericDType::U8,
        NumericElementType::U16 => NumericDType::U16,
        NumericElementType::U32 => NumericDType::U32,
        NumericElementType::U64 => NumericDType::U64,
    };
    if logical
        && (storage != GpuTensorStorage::Real
            || !matches!(
                dtype,
                NumericDType::U8 | NumericDType::F32 | NumericDType::F64
            ))
    {
        return Err(invalid(
            "selected logical readback has incompatible native storage",
        ));
    }
    let shape = handle.shape();
    let rank = shape.len();
    let footprint = ProviderQueryMetadataFootprint {
        operation: "transfer.download",
        input_ranks: &[rank],
        output_ranks: &[rank],
        preparation_shape_ranks: &[rank],
        preparation_representation_slots: 0,
    };
    let query_bytes = footprint
        .requested_bytes()
        .ok_or_else(|| invalid("selected host query extent overflows"))?;
    let shape_bytes = rank
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("selected host shape extent overflows"))?;
    let payload_bytes = checked_numeric_download_host_payload_bytes(handle)
        .ok_or_else(|| invalid("selected host payload extent overflows"))?;
    let ResourceByteBound::Known(host_workspace) = workspace_bound else {
        return Err(invalid(
            "selected provider has no finite native download bound",
        ));
    };
    if payload_bytes
        .checked_add(shape_bytes)
        .is_none_or(|minimum| host_workspace < minimum)
    {
        return Err(invalid(
            "selected provider host workspace omits returned backing",
        ));
    }
    if !matches!(scratch_bound, ResourceByteBound::Known(_)) {
        return Err(invalid("selected provider has no finite staging bound"));
    }
    let ResourceByteBound::Known(callback_bytes) = callback_bound else {
        return Err(invalid("selected provider has no finite callback bound"));
    };
    let count = shape
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invalid("selected host value extent overflows"))?;
    let adopted_raw = match (storage, logical) {
        (GpuTensorStorage::Real, true) if dtype != NumericDType::U8 => {
            ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(count, rank)
        }
        (GpuTensorStorage::Real, true) => {
            ExactHostLogicalTensorAllocation::adopted_requested_bytes(count, rank)
        }
        (GpuTensorStorage::Real, false) => {
            ExactHostTensorAllocation::adopted_requested_bytes(dtype, count, rank)
        }
        (GpuTensorStorage::ComplexInterleaved, false) if dtype.integer_class().is_some() => {
            ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                dtype, count, rank,
            )
        }
        (GpuTensorStorage::ComplexInterleaved, false) => {
            ExactHostComplexTensorAllocation::adopted_requested_bytes(
                dtype,
                count
                    .checked_mul(2)
                    .ok_or_else(|| invalid("selected complex extent overflows"))?,
                rank,
            )
        }
        _ => return Err(invalid("selected native layout is unsupported")),
    }
    .ok()
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or_else(|| invalid("selected host value extent overflows"))?;
    let converting = (storage == GpuTensorStorage::ComplexInterleaved
        && dtype.integer_class().is_some())
        || (logical && dtype != NumericDType::U8);
    let controls = HostAllocationPlan::retained_exact_output_bytes(
        0,
        if storage == GpuTensorStorage::ComplexInterleaved && dtype.integer_class().is_some() {
            ExactHostIntegerComplexTensorAllocation::retained_claim_count()
        } else {
            ExactHostTensorAllocation::retained_claim_count()
        },
    )
    .map_err(|_| invalid("selected host claim extent overflows"))?;
    let exact_output = adopted_raw
        .checked_add(controls)
        .ok_or_else(|| invalid("selected host output extent overflows"))?;
    let retained_value_upper = if converting {
        0
    } else {
        let overhead = adopted_raw
            .checked_sub(payload_bytes)
            .and_then(|bytes| bytes.checked_sub(shape_bytes))
            .ok_or_else(|| invalid("selected host wrapper extent overflows"))?;
        host_workspace
            .checked_add(overhead)
            .and_then(|bytes| bytes.checked_add(controls))
            .ok_or_else(|| invalid("selected host retained extent overflows"))?
    };
    let active_upper = retained_value_upper
        .checked_add(if converting { host_workspace } else { 0 })
        .and_then(|bytes| bytes.checked_add(query_bytes.checked_mul(2)?))
        .and_then(|bytes| bytes.checked_add(shape_bytes.checked_mul(2)?))
        .ok_or_else(|| invalid("selected host readback peak overflows"))?;
    Ok(ExactNativeDownloadQuote {
        dtype,
        storage,
        logical,
        host_workspace,
        payload_bytes,
        query_bytes,
        shape_bytes,
        callback_bytes,
        retained_value_upper,
        active_upper,
        precharged_output_bytes: if converting { exact_output } else { 0 },
    })
}
