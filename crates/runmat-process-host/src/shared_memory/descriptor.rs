use serde::{Deserialize, Serialize};

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedMemoryKind {
    FileBacked,
    #[cfg(unix)]
    UnixFileDescriptor,
    #[cfg(windows)]
    WindowsHandle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedMemoryDescriptor {
    pub kind: SharedMemoryKind,
    pub name: String,
    pub byte_length: u64,
    pub nonce: [u8; 16],
}

impl SharedMemoryDescriptor {
    pub fn validate(&self) -> ProcessHostResult<()> {
        if self.name.is_empty() {
            return Err(ProcessHostError::Configuration(
                "shared-memory name must not be empty".into(),
            ));
        }
        if self.byte_length == 0 {
            return Err(ProcessHostError::Configuration(
                "shared-memory length must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}
