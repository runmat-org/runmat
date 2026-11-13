//! Bounded typed projection of a native-owned prepared-library I/O cause.
//!
//! The child and driver run on the same host. Preserve the actual OS error
//! code and path, never infer a cause from a message or public identifier.

use super::*;
use crate::foreign::native_ffi::adapter::PreparedLibraryReadError;
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "kind")]
pub(in crate::foreign::native_ffi::isolation) enum NativeWireCause {
    PreparedLibraryRead { path: PathBuf, raw_os_error: i32 },
}

impl NativeWireCause {
    pub(super) fn from_runtime(error: &crate::RuntimeError) -> Option<Self> {
        let source =
            std::error::Error::source(error)?.downcast_ref::<PreparedLibraryReadError>()?;
        Some(Self::PreparedLibraryRead {
            path: source.path.clone(),
            raw_os_error: source.source.raw_os_error()?,
        })
    }

    pub(super) fn into_source(self) -> PreparedLibraryReadError {
        match self {
            Self::PreparedLibraryRead { path, raw_os_error } => PreparedLibraryReadError {
                path,
                source: std::io::Error::from_raw_os_error(raw_os_error),
            },
        }
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        match self {
            Self::PreparedLibraryRead { path, .. } => {
                let path = path
                    .to_str()
                    .ok_or("prepared-library error path is not UTF-8")?;
                validate_text(path, "prepared-library error path")
            }
        }
    }
}

#[cfg(test)]
mod tests;
