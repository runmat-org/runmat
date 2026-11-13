//! Plot-source export failures, retaining readback causes across presentation layers.
use crate::gpu::PlotReadbackError;
use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlotDataExportError {
    SourceUnavailable(&'static str),
    InvalidSourceLayout(String),
    Readback(PlotReadbackError),
}

impl fmt::Display for PlotDataExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceUnavailable(message) => f.write_str(message),
            Self::InvalidSourceLayout(message) => f.write_str(message),
            Self::Readback(error) => write!(f, "{error}"),
        }
    }
}

impl Error for PlotDataExportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Readback(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PlotReadbackError> for PlotDataExportError {
    fn from(error: PlotReadbackError) -> Self {
        Self::Readback(error)
    }
}
