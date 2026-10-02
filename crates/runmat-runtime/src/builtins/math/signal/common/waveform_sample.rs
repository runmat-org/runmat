//! Carrier admission for real-valued pulse and waveform samples.

use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WaveformSampleError {
    Complex,
    Unsupported,
}

/// Inspect only authoritative host value kinds and resident metadata. This
/// runs before a deferred control or sample can access its provider.
pub(crate) fn admit_real_waveform_sample(value: &Value) -> Result<(), WaveformSampleError> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::Tensor(_) => Ok(()),
        Value::GpuTensor(handle) if handle_storage(handle) == GpuTensorStorage::Real => Ok(()),
        Value::GpuTensor(_) | Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(WaveformSampleError::Complex)
        }
        _ => Err(WaveformSampleError::Unsupported),
    }
}
