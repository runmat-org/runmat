//! Structured scope-limit failures, available through the original I/O cause.
use std::{fmt, io};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceResourceLimit {
    pub required_bytes: u64,
    pub maximum_bytes: u64,
}
impl fmt::Display for WorkspaceResourceLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workspace requests {} bytes with {} admitted",
            self.required_bytes, self.maximum_bytes
        )
    }
}
impl std::error::Error for WorkspaceResourceLimit {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn exceeded(required_bytes: u64, maximum_bytes: u64) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        WorkspaceResourceLimit {
            required_bytes,
            maximum_bytes,
        },
    )
}

pub fn resource_limit(error: &io::Error) -> Option<&WorkspaceResourceLimit> {
    error.get_ref()?.downcast_ref()
}
