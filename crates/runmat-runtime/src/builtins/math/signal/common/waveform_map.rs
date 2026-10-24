//! Typed host waveform mapping with fallible output storage and invocation cooperation.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{NumericStorage, Tensor};

use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

use super::SignalCooperation;

// Limits uninterrupted coefficient work, not the execution service's yield policy.
const MAP_CHUNK: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WaveformOutputMode {
    PreserveSingle,
    Double,
}

#[derive(Debug)]
pub(crate) enum WaveformMapError {
    Storage(String),
    Allocation(std::collections::TryReserveError),
    Tensor(String),
}

impl WaveformMapError {
    pub(crate) fn into_runtime_error(
        self,
        builtin: &'static str,
        descriptor: &'static BuiltinErrorDescriptor,
    ) -> RuntimeError {
        let detail = match self {
            Self::Storage(detail) | Self::Tensor(detail) => detail,
            Self::Allocation(cause) => {
                let mut builder = build_runtime_error(format!("{}: {cause}", descriptor.message))
                    .with_builtin(builtin)
                    .with_source(cause);
                if let Some(identifier) = descriptor.identifier {
                    builder = builder.with_identifier(identifier);
                }
                let mut error = builder.build();
                error.gpu_gather_retry = GpuGatherRetry::Never;
                return error;
            }
        };
        let mut builder =
            build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(builtin);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        let mut error = builder.build();
        error.gpu_gather_retry = GpuGatherRetry::Never;
        error
    }
}

macro_rules! waveform_try {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        }
    };
}

pub(crate) async fn map_real_waveform(
    tensor: Tensor,
    mode: WaveformOutputMode,
    evaluate: impl Fn(f64) -> f64,
    cooperation: &mut SignalCooperation,
) -> crate::BuiltinResult<Result<Tensor, WaveformMapError>> {
    cooperation.completed(0).await?;
    let shape = tensor.shape().to_vec();
    let storage = waveform_try!(tensor
        .into_numeric_storage()
        .map_err(WaveformMapError::Storage));
    let output = match (storage, mode) {
        (NumericStorage::F32(values), WaveformOutputMode::PreserveSingle) => {
            NumericStorage::F32(waveform_try!(
                map_values(
                    values,
                    |value| f64::from(value),
                    &evaluate,
                    |value| value as f32,
                    cooperation,
                )
                .await?
            ))
        }
        (NumericStorage::F64(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                std::convert::identity,
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::F32(values), WaveformOutputMode::Double) => {
            NumericStorage::F64(waveform_try!(
                map_values(
                    values,
                    |value| f64::from(value),
                    &evaluate,
                    std::convert::identity,
                    cooperation,
                )
                .await?
            ))
        }
        (NumericStorage::I8(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::I16(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::I32(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::I64(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| value as f64,
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::U8(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::U16(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::U32(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| f64::from(value),
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
        (NumericStorage::U64(values), _) => NumericStorage::F64(waveform_try!(
            map_values(
                values,
                |value| value as f64,
                &evaluate,
                std::convert::identity,
                cooperation,
            )
            .await?
        )),
    };
    Ok(Tensor::from_numeric_storage(output, shape)
        .map_err(|error| WaveformMapError::Tensor(error.to_string())))
}

async fn map_values<Input: Copy, Output>(
    values: Vec<Input>,
    to_double: impl Fn(Input) -> f64,
    evaluate: &impl Fn(f64) -> f64,
    cast: impl Fn(f64) -> Output,
    cooperation: &mut SignalCooperation,
) -> crate::BuiltinResult<Result<Vec<Output>, WaveformMapError>> {
    let mut output = Vec::new();
    waveform_try!(output
        .try_reserve_exact(values.len())
        .map_err(WaveformMapError::Allocation));
    for chunk in values.chunks(MAP_CHUNK) {
        for &value in chunk {
            output.push(cast(evaluate(to_double(value))));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Ok(output))
}

#[cfg(test)]
#[path = "waveform_map/tests.rs"]
mod tests;
