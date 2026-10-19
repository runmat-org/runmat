//! Adopt provider-owned results without mutating borrowed views or producer facts.
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

/// Preserve existing intent, supplying Automatic only when it was unspecified.
/// View creation remains fallible: quarantined owners cannot publish a new view.
pub fn resident_gpu_value(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let handle = if runmat_accelerate_api::handle_provenance(&handle).is_none() {
        match handle
            .with_provenance(GpuHandleProvenance::Automatic)
            .map_err(|error| {
                build_runtime_error(format!("cannot adopt resident provider result: {error}"))
                    .with_source(error)
                    .build()
            }) {
            Ok(adopted) => adopted,
            Err(error) => return Err(super::reject_gpu_output(handle, error)),
        }
    } else {
        if let Err(source) = handle.allocation_lease().validate_live() {
            let error =
                build_runtime_error(format!("cannot adopt resident provider result: {source}"))
                    .with_source(source)
                    .build();
            return Err(super::reject_gpu_output(handle, error));
        }
        handle
    };
    runmat_accelerate_api::mark_residency(&handle);
    Ok(Value::GpuTensor(handle))
}

/// Logical interpretation must have been admitted by the producer's owner.
pub fn logical_gpu_value(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if !runmat_accelerate_api::handle_is_logical(&handle)
        || handle.physical().storage() != GpuTensorStorage::Real
    {
        return Err(super::reject_gpu_output(
            handle,
            build_runtime_error("provider result was not produced as a real logical array").build(),
        ));
    }
    resident_gpu_value(handle)
}

/// A complex wrapper cannot promote real physical storage to interleaved data.
pub fn complex_gpu_value(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(&handle)
        || handle.physical().storage() != GpuTensorStorage::ComplexInterleaved
    {
        return Err(super::reject_gpu_output(
            handle,
            build_runtime_error("provider result does not own complex interleaved storage").build(),
        ));
    }
    resident_gpu_value(handle)
}

/// Explicit input intent dominates automatic intent. Validate the live lease
/// even when the existing view already has the requested provenance.
pub(crate) fn propagate_output_provenance<'a>(
    output: &mut GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> BuiltinResult<()> {
    let provenance = if inputs
        .into_iter()
        .any(runmat_accelerate_api::handle_is_explicit)
    {
        GpuHandleProvenance::Explicit
    } else {
        GpuHandleProvenance::Automatic
    };
    if runmat_accelerate_api::handle_provenance(output) != Some(provenance) {
        *output = output.with_provenance(provenance).map_err(|error| {
            build_runtime_error(format!("cannot propagate resident output intent: {error}"))
                .with_source(error)
                .build()
        })?;
    } else {
        output
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                build_runtime_error(format!("cannot propagate resident output intent: {source}"))
                    .with_source(source)
                    .build()
            })?;
    }
    Ok(())
}

/// Preserve intent on an owned result, retaining physical retirement failures
/// if its validated publication view cannot be created.
pub(crate) fn propagate_owned_output_provenance<'a>(
    mut output: GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> BuiltinResult<GpuTensorHandle> {
    match propagate_output_provenance(&mut output, inputs) {
        Ok(()) => Ok(output),
        Err(error) => Err(super::reject_gpu_output(output, error)),
    }
}

#[cfg(test)]
mod tests;
