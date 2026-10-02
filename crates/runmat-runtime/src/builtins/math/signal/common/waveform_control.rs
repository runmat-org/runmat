//! Scalar waveform controls: host-visible admission precedes device access.

use runmat_accelerate_api::{handle_storage, GpuTensorStorage};
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, tensor, validation};
use crate::RuntimeError;

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
        extension: &'static BuiltinExtensionDescriptor,
        builtin: &'static str,
        role: &'static str,
    },
    Resident {
        value: &'a Value,
        extension: &'static BuiltinExtensionDescriptor,
        builtin: &'static str,
        role: &'static str,
    },
}

impl<'a> PreparedWaveformControl<'a> {
    /// Prove a host control's integer boundary before any resident argument is
    /// downloaded. Resident controls remain deferred to `resolve`.
    pub(crate) async fn admit_host_boundary(&self) -> Result<(), WaveformControlError> {
        if let Self::Host {
            value,
            extension,
            builtin,
            role,
            ..
        } = self
        {
            validation::ensure_runmat_integer_f64_boundary(value, extension, builtin, role)
                .await
                .map_err(WaveformControlError::Boundary)?;
        }
        Ok(())
    }

    pub(crate) fn host_value(&self) -> Option<f64> {
        match self {
            Self::Host { scalar, .. } => Some(*scalar),
            Self::Resident { .. } => None,
        }
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
                    builtin,
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
                    extension,
                    builtin,
                    role,
                })
            }
        }
    }

    pub(crate) async fn resolve(self) -> Result<f64, WaveformControlError> {
        match self {
            Self::Host {
                value,
                scalar,
                extension,
                builtin,
                role,
            } => {
                validation::ensure_runmat_integer_f64_boundary(value, extension, builtin, role)
                    .await
                    .map_err(WaveformControlError::Boundary)?;
                Ok(scalar)
            }
            Self::Resident {
                value,
                extension,
                builtin,
                role,
            } => {
                if validation::value_has_native_integer_class(value) {
                    crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
                        .map_err(WaveformControlError::Boundary)?;
                }
                let host = gpu_helpers::gather_value_async(value)
                    .await
                    .map_err(WaveformControlError::Provider)?;
                validation::ensure_runmat_integer_f64_boundary(&host, extension, builtin, role)
                    .await
                    .map_err(WaveformControlError::Boundary)?;
                tensor::scalar_f64_from_host_value(&host)
                    .map_err(WaveformControlError::Invalid)?
                    .ok_or_else(|| {
                        WaveformControlError::Invalid("expected a numeric scalar".into())
                    })
            }
        }
    }
}
