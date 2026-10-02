//! Scalar waveform controls: host-visible admission precedes device access.

use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, tensor, validation};
use crate::RuntimeError;

use super::{admit_resident_waveform_operand, prepared_real_integer_tensor, SignalCooperation};

#[derive(Debug)]
pub(crate) enum WaveformControlError {
    Invalid(String),
    Boundary(RuntimeError),
    Provider(RuntimeError),
}

/// A resident control is deferred until all host-visible arguments have passed
/// admission. The downloaded value is used for both exactness and scalar decode.
pub(crate) enum PreparedWaveformControl<'a> {
    Host {
        value: &'a Value,
        scalar: f64,
        boundary_admitted: bool,
        extension: &'static BuiltinExtensionDescriptor,
        role: &'static str,
    },
    Resident {
        value: &'a Value,
        extension: &'static BuiltinExtensionDescriptor,
        role: &'static str,
    },
}

impl<'a> PreparedWaveformControl<'a> {
    /// Prove a host control's integer boundary before any resident argument is
    /// downloaded. Resident controls remain deferred to `resolve`.
    pub(crate) async fn admit_host_boundary(
        &mut self,
        cooperation: &mut SignalCooperation,
    ) -> Result<(), WaveformControlError> {
        if let Self::Host {
            value,
            extension,
            role,
            boundary_admitted,
            ..
        } = self
        {
            cooperation
                .prepare_integer_boundary(value, extension, role)
                .await
                .map_err(WaveformControlError::Boundary)?;
            *boundary_admitted = true;
        }
        Ok(())
    }

    pub(crate) fn host_value(&self) -> Option<f64> {
        match self {
            Self::Host { scalar, .. } => Some(*scalar),
            Self::Resident { .. } => None,
        }
    }

    /// Apply host-visible resident metadata checks only after every ordinary
    /// host argument has passed admission, but before any sample download.
    pub(crate) fn admit_resident_metadata(
        &self,
        builtin: &'static str,
        internal: &'static BuiltinErrorDescriptor,
    ) -> Result<(), WaveformControlError> {
        let Self::Resident {
            value, extension, ..
        } = self
        else {
            return Ok(());
        };
        if validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
                .map_err(WaveformControlError::Boundary)?;
        }
        admit_resident_waveform_operand(value, builtin, "control", internal)
            .map_err(WaveformControlError::Boundary)
    }

    pub(crate) fn admit(
        value: &'a Value,
        extension: &'static BuiltinExtensionDescriptor,
        builtin: &'static str,
        role: &'static str,
    ) -> Result<Self, WaveformControlError> {
        validation::reject_typed_complex_integer(value, builtin)
            .map_err(WaveformControlError::Boundary)?;
        match value {
            Value::GpuTensor(handle) => {
                let count = handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, &extent| count.checked_mul(extent));
                if count != Some(1) || handle_storage(handle) != GpuTensorStorage::Real {
                    return Err(WaveformControlError::Invalid(
                        "expected a real scalar gpuArray".into(),
                    ));
                }
                Ok(Self::Resident {
                    value,
                    extension,
                    role,
                })
            }
            _ => {
                let scalar = tensor::scalar_f64_from_host_value(value)
                    .map_err(WaveformControlError::Invalid)?
                    .ok_or_else(|| {
                        WaveformControlError::Invalid("expected a numeric scalar".into())
                    })?;
                Ok(Self::Host {
                    value,
                    scalar,
                    boundary_admitted: false,
                    extension,
                    role,
                })
            }
        }
    }

    pub(crate) async fn resolve(
        self,
        cooperation: &mut SignalCooperation,
    ) -> Result<f64, WaveformControlError> {
        match self {
            Self::Host {
                value,
                scalar,
                extension,
                role,
                boundary_admitted,
                ..
            } => {
                if !boundary_admitted {
                    cooperation
                        .prepare_integer_boundary(value, extension, role)
                        .await
                        .map_err(WaveformControlError::Boundary)?;
                }
                Ok(scalar)
            }
            Self::Resident {
                value,
                extension,
                role,
                ..
            } => {
                let prepared = cooperation
                    .prepare_integer_boundary(value, extension, role)
                    .await
                    .map_err(WaveformControlError::Boundary)?;
                let payload = prepared.into_resident_payload();
                let host = if let Some(tensor) =
                    prepared_real_integer_tensor(payload).map_err(WaveformControlError::Invalid)?
                {
                    Value::Tensor(tensor)
                } else {
                    gpu_helpers::gather_value_async(value)
                        .await
                        .map_err(WaveformControlError::Provider)?
                };
                tensor::scalar_f64_from_host_value(&host)
                    .map_err(WaveformControlError::Invalid)?
                    .ok_or_else(|| {
                        WaveformControlError::Invalid("expected a numeric scalar".into())
                    })
            }
        }
    }
}
