//! Consume a validated native integer readback without repeating a transfer.

use runmat_value::{Tensor, Value};

use runmat_builtins::BuiltinErrorDescriptor;

use crate::builtins::common::validation::PreparedIntegerPayload;
use crate::builtins::common::{gpu_helpers, tensor, validation};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::SignalCooperation;

const LOGICAL_CONVERSION_CHUNK: usize = 4096;

#[derive(Debug)]
pub(crate) enum WaveformInputError {
    Invalid(String),
    Allocation(std::collections::TryReserveError),
    Tensor(String),
}

impl WaveformInputError {
    pub(crate) fn into_runtime_error(
        self,
        builtin: &'static str,
        invalid: &'static BuiltinErrorDescriptor,
        internal: &'static BuiltinErrorDescriptor,
    ) -> RuntimeError {
        let (descriptor, detail, allocation) = match self {
            Self::Invalid(detail) => (invalid, detail, None),
            Self::Tensor(detail) => (internal, detail, None),
            Self::Allocation(cause) => (internal, cause.to_string(), Some(cause)),
        };
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(builtin)
            .with_gpu_gather_retry(GpuGatherRetry::Never);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        let mut error = builder.build();
        if let Some(cause) = allocation {
            error.source = Some(Box::new(cause));
        }
        error
    }
}

/// Convert host logical samples through the same invocation budget as the
/// numeric mapper. Other already-typed values retain the existing conversion.
pub(crate) async fn waveform_input_tensor(
    value: Value,
    builtin: &'static str,
    cooperation: &mut SignalCooperation,
) -> crate::BuiltinResult<Result<Tensor, WaveformInputError>> {
    let array = match value {
        Value::LogicalArray(array) => array,
        other => {
            return Ok(
                tensor::value_into_tensor_for(builtin, other).map_err(WaveformInputError::Invalid)
            );
        }
    };
    cooperation.completed(0).await?;
    let mut data = Vec::new();
    if let Err(cause) = data.try_reserve_exact(array.data.len()) {
        return Ok(Err(WaveformInputError::Allocation(cause)));
    }
    for chunk in array.data.chunks(LOGICAL_CONVERSION_CHUNK) {
        data.extend(chunk.iter().map(|&bit| if bit != 0 { 1.0 } else { 0.0 }));
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Tensor::new(data, array.shape).map_err(WaveformInputError::Tensor))
}

/// Authenticate a resident waveform operand before another operand can cause
/// a transfer. Class/owner admission and the physical lease remain separate
/// from the integer exactness scan.
pub(crate) fn admit_resident_waveform_operand(
    value: &Value,
    builtin: &'static str,
    role: &str,
    internal: &'static BuiltinErrorDescriptor,
) -> Result<(), RuntimeError> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    validation::validate_resident_metadata(value).map_err(|source| {
        let span = source.span;
        let mut builder = build_runtime_error(format!(
            "{}: resident {role} metadata is invalid: {}",
            internal.message,
            source.message()
        ))
        .with_builtin(builtin)
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never);
        if let Some(identifier) = internal.identifier {
            builder = builder.with_identifier(identifier);
        }
        let mut owner = builder.build();
        owner.span = span;
        owner
    })?;
    handle
        .allocation_lease()
        .validate_live()
        .map_err(|source| {
            build_runtime_error(format!(
                "{builtin}: resident {role} allocation is unavailable: {source}"
            ))
            .with_builtin(builtin)
            .with_source(source)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build()
        })?;
    handle.view().validate(handle.physical()).map_err(|source| {
        build_runtime_error(format!(
            "{builtin}: resident {role} view is invalid: {source}"
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
