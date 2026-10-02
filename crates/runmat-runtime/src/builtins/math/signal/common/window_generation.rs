//! One mathematical window generator with invocation-owned CPU cooperation.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{NumericStorage, Tensor, Value};

use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::{SignalCooperation, WindowArgError, WindowOptions, WindowOutputType, WindowSampling};

// Bounded work between calls to the execution-owned budget, not a scheduling
// threshold. The execution service alone decides whether a task turn is due.
const COEFFICIENT_CHUNK: usize = 4096;

pub(crate) fn window_allocation_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    cause: std::collections::TryReserveError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {cause}", descriptor.message))
        .with_builtin(builtin)
        .with_source(cause);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut error = builder.build();
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(crate) async fn window_tensor(
    options: WindowOptions,
    coeff: impl Fn(usize, usize) -> f64,
    cooperation: &mut SignalCooperation,
) -> Result<Value, WindowArgError> {
    cooperation
        .completed(0)
        .await
        .map_err(WindowArgError::Execution)?;
    let len = options.len;
    if len == 0 {
        let storage = match options.output_type {
            WindowOutputType::Double => NumericStorage::F64(Vec::new()),
            WindowOutputType::Single => NumericStorage::F32(Vec::new()),
        };
        return Tensor::from_numeric_storage(storage, vec![0, 1])
            .map(Value::Tensor)
            .map_err(|e| WindowArgError::TensorBuild(e.to_string()));
    }
    if len == 1 {
        let storage = match options.output_type {
            WindowOutputType::Double => NumericStorage::F64(vec![1.0]),
            WindowOutputType::Single => NumericStorage::F32(vec![1.0]),
        };
        return Tensor::from_numeric_storage(storage, vec![1, 1])
            .map(Value::Tensor)
            .map_err(|e| WindowArgError::TensorBuild(e.to_string()));
    }
    let effective_len = match options.sampling {
        WindowSampling::Symmetric => len,
        WindowSampling::Periodic => len.checked_add(1).ok_or(WindowArgError::InvalidLength)?,
    };
    let storage = match options.output_type {
        WindowOutputType::Double => NumericStorage::F64(
            populate(
                len,
                effective_len,
                &coeff,
                std::convert::identity,
                cooperation,
            )
            .await?,
        ),
        WindowOutputType::Single => NumericStorage::F32(
            populate(
                len,
                effective_len,
                &coeff,
                |value| value as f32,
                cooperation,
            )
            .await?,
        ),
    };
    Tensor::from_numeric_storage(storage, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|e| WindowArgError::TensorBuild(e.to_string()))
}

async fn populate<T>(
    len: usize,
    effective_len: usize,
    coeff: &impl Fn(usize, usize) -> f64,
    cast: impl Fn(f64) -> T,
    cooperation: &mut SignalCooperation,
) -> Result<Vec<T>, WindowArgError> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(WindowArgError::Allocation)?;
    for start in (0..len).step_by(COEFFICIENT_CHUNK) {
        let end = start.saturating_add(COEFFICIENT_CHUNK).min(len);
        for idx in start..end {
            data.push(cast(coeff(idx, effective_len)));
        }
        cooperation
            .completed((end - start) as u64)
            .await
            .map_err(WindowArgError::Execution)?;
    }
    Ok(data)
}

#[cfg(test)]
#[path = "window_generation/tests.rs"]
mod tests;
