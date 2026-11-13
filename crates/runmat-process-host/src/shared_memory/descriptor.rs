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
    pub sha256: [u8; 32],
}

impl SharedMemoryDescriptor {
    pub fn validate(&self) -> ProcessHostResult<()> {
        if self.name.len() != 32
            || !self.name.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.name != hex_nonce(self.nonce)
        {
            return Err(ProcessHostError::Configuration(
                "shared-memory name must be the descriptor nonce in hexadecimal".into(),
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

pub(crate) fn hex_nonce(nonce: [u8; 16]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(32);
    for byte in nonce {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}
