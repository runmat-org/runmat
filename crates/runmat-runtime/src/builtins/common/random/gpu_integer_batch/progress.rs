//! Authenticate the provisional output and acceptance-mask allocations.
use super::error;
use crate::builtins::common::{
    gpu_helpers, random_args::validate_constructor_gpu_storage_protecting,
};
use crate::BuiltinResult;
use runmat_accelerate_api::random::{
    GpuIntegerBatchProgress, GpuIntegerBatchRequest, GpuIntegerPending,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

/// Authenticate both provisional allocations before acknowledging a batch;
/// only the final batch's numeric output receives user provenance.
pub(super) fn validate_progress(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    request: GpuIntegerBatchRequest<'_>,
    progress: GpuIntegerBatchProgress,
    output_bytes: u64,
    accepted_bytes: u64,
) -> BuiltinResult<(GpuIntegerPending, bool)> {
    let count = request
        .validate()
        .ok_or_else(|| error("randi: selected batch request is no longer valid"))?;
    let previous = request.pending;
    let maximum_remaining = previous.map_or_else(
        || u64::try_from(count).expect("validated integer request count fits u64"),
        |pending| pending.remaining,
    );
    let (buffers, complete) = match progress {
        GpuIntegerBatchProgress::Pending(buffers) => (buffers, false),
        GpuIntegerBatchProgress::Complete(buffers) => (buffers, true),
    };
    let output = match validate_constructor_gpu_storage_protecting(
        "randi",
        provider,
        buffers.output,
        request.integer.shape,
        GpuTensorStorage::Real,
        request.integer.class.precision(),
        request.integer.class.integer_type(),
        false,
        &[source],
    ) {
        Ok(output) => output,
        Err(cause) => return Err(gpu_helpers::reject_gpu_output(buffers.accepted, cause)),
    };
    let accepted = match validate_constructor_gpu_storage_protecting(
        "randi",
        provider,
        buffers.accepted,
        request.integer.shape,
        GpuTensorStorage::Real,
        None,
        Some(runmat_accelerate_api::IntegerElementType::U32),
        false,
        &[source, &output],
    ) {
        Ok(accepted) => accepted,
        Err(cause) => return Err(gpu_helpers::reject_gpu_output(output, cause)),
    };
    if previous.is_some_and(|previous| {
        !output.shares_allocation_with(&previous.output)
            || !accepted.shares_allocation_with(&previous.accepted)
    }) || output.physical().requested_bytes() != output_bytes
        || accepted.physical().requested_bytes() != accepted_bytes
        || buffers.remaining > maximum_remaining
        || (complete && buffers.remaining != 0)
        || (!complete && buffers.remaining == 0)
    {
        return Err(gpu_helpers::reject_gpu_outputs(
            [output, accepted],
            error("randi: batch output disagrees with physical proposal"),
        ));
    }
    Ok((
        GpuIntegerPending {
            output,
            accepted,
            remaining: buffers.remaining,
        },
        complete,
    ))
}

#[cfg(test)]
#[path = "progress/tests.rs"]
mod tests;
