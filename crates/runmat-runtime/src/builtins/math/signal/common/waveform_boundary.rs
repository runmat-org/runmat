//! Consume a validated native integer readback without repeating a transfer.

use runmat_value::{Tensor, Value};

use crate::builtins::common::validation::PreparedIntegerPayload;
use crate::builtins::common::{gpu_helpers, validation};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

/// Authenticate a resident waveform operand before another operand can cause
/// a transfer. Class/owner admission and the physical lease remain separate
/// from the integer exactness scan.
pub(crate) fn admit_resident_waveform_operand(
    value: &Value,
    builtin: &'static str,
    role: &str,
) -> Result<(), RuntimeError> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    validation::validate_resident_metadata(value)?;
    handle.allocation_lease().validate_live().map_err(|source| {
        build_runtime_error(format!(
            "{builtin}: resident {role} allocation is unavailable: {source}"
        ))
        .with_builtin(builtin)
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
    })
}

pub(crate) fn prepared_real_integer_tensor(
    payload: Option<PreparedIntegerPayload>,
) -> Result<Option<Tensor>, String> {
    match payload {
        None => Ok(None),
        Some(PreparedIntegerPayload::Real(gathered)) => Tensor::new_integer(
            gpu_helpers::integer_storage_from_owned(gathered.data),
            gathered.shape,
        )
        .map(Some),
        Some(PreparedIntegerPayload::Complex(_)) => {
            Err("complex integer samples require owning arithmetic rejection".into())
        }
    }
}
