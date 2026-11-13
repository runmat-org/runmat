//! Typed causes for GPU readback, including failures during mapping cleanup.
use runmat_accelerate_api::WgpuAccessError;
use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlotReadbackError {
    ByteSizeOverflow,
    BufferTooSmall {
        requested: usize,
        available: u64,
    },
    CopyBufferTooSmall {
        requested: usize,
        available: u64,
    },
    Access(WgpuAccessError),
    MapCallbackDropped,
    Map(wgpu::BufferAsyncError),
    Decode(&'static str),
    CleanupAccess(WgpuAccessError),
    UnmapPanicked,
    WithCleanup {
        primary: Box<Self>,
        cleanup: Box<Self>,
    },
}

impl PlotReadbackError {
    pub fn cleanup_cause(&self) -> Option<&Self> {
        match self {
            Self::WithCleanup { cleanup, .. } => Some(cleanup),
            _ => None,
        }
    }
}

impl fmt::Display for PlotReadbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteSizeOverflow => f.write_str("readback byte size overflow"),
            Self::BufferTooSmall { .. } => f.write_str("GPU readback buffer too small"),
            Self::CopyBufferTooSmall { .. } => f.write_str("GPU copy readback buffer too small"),
            Self::Access(error) => write!(f, "{error}"),
            Self::MapCallbackDropped => f.write_str("map callback dropped"),
            Self::Map(error) => write!(f, "{error}"),
            Self::Decode(message) => f.write_str(message),
            Self::CleanupAccess(error) => {
                write!(
                    f,
                    "GPU readback cleanup could not acquire its device: {error}"
                )
            }
            Self::UnmapPanicked => {
                f.write_str("GPU readback unmap panicked; mapping state is uncertain")
            }
            Self::WithCleanup { primary, cleanup } => {
                write!(f, "{primary}; additionally, {cleanup}")
            }
        }
    }
}

impl Error for PlotReadbackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Access(error) | Self::CleanupAccess(error) => Some(error),
            Self::Map(error) => Some(error),
            Self::WithCleanup { primary, .. } => Some(primary.as_ref()),
            _ => None,
        }
    }
}
