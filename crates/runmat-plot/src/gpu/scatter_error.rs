//! Typed failures before a scatter GPU pack publishes vertices.

use super::ScalarType;
use runmat_accelerate_api::WgpuAccessError;
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterPackError {
    EmptyInputs,
    RetainedAttributeMismatch,
    SourceAttributeMismatch,
    MissingNativeF64Features {
        required: wgpu::Features,
        available: wgpu::Features,
    },
    BufferAccess(WgpuAccessError),
}

impl fmt::Display for ScatterPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInputs => f.write_str("scatter GPU inputs are empty"),
            Self::RetainedAttributeMismatch => {
                f.write_str("retained scatter attribute units or layout differ from pack inputs")
            }
            Self::SourceAttributeMismatch => {
                f.write_str("scatter attribute units or layout differ from source buffers")
            }
            Self::MissingNativeF64Features { required, available } => write!(
                f,
                "scatter binary64 input requires device features {required:?}; available {available:?}"
            ),
            Self::BufferAccess(error) => write!(f, "scatter GPU input buffer unavailable: {error}"),
        }
    }
}

impl Error for ScatterPackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BufferAccess(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WgpuAccessError> for ScatterPackError {
    fn from(error: WgpuAccessError) -> Self {
        Self::BufferAccess(error)
    }
}

pub(super) fn preflight_native_f64_features(
    scalar: ScalarType,
    available: wgpu::Features,
) -> Result<(), ScatterPackError> {
    if scalar == ScalarType::F64 {
        let required = runmat_accelerate_api::native_wgpu_f64::REQUIRED_FEATURES;
        if !available.contains(required) {
            return Err(ScatterPackError::MissingNativeF64Features {
                required,
                available,
            });
        }
    }
    Ok(())
}
