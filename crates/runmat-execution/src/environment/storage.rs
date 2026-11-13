//! Exact logical data and encoded bytes of the original snapshot, without a wire copy.
//! This component is a minimum reservation, not allocator overhead or process RSS.
use super::*;
use std::io::{self, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnvironmentStorageProfile {
    pub entries: u64,
    pub utf8_bytes: u64,
    pub typed_data_bytes: u64,
    pub json_bytes: u64,
}

impl EnvironmentStorageProfile {
    /// One original typed snapshot plus its complete JSON transport buffer.
    /// Additional copies, runtime maps and execution storage belong to their owners.
    pub fn minimum_memory_bytes(self) -> Result<u64, ContractError> {
        self.typed_data_bytes
            .checked_add(self.json_bytes)
            .ok_or_else(overflow)
    }
}

impl EnvironmentSnapshot {
    pub fn storage_profile(&self) -> Result<EnvironmentStorageProfile, ContractError> {
        self.validate()?;
        let entries = u64::try_from(self.entries.len()).map_err(|_| overflow())?;
        let utf8_bytes = self
            .entries
            .iter()
            .try_fold(0_u64, |bytes, (key, value)| {
                bytes
                    .checked_add(key.len() as u64)?
                    .checked_add(value.len() as u64)
            })
            .ok_or_else(overflow)?;
        let typed_data_bytes = entries
            .checked_mul(std::mem::size_of::<(String, String)>() as u64)
            .and_then(|bytes| bytes.checked_add(utf8_bytes))
            .ok_or_else(overflow)?;
        let mut count = Count(0);
        serde_json::to_writer(&mut count, self).map_err(|_| overflow())?;
        Ok(EnvironmentStorageProfile {
            entries,
            utf8_bytes,
            typed_data_bytes,
            json_bytes: count.0,
        })
    }
}

struct Count(u64);
impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("environment transport size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn overflow() -> ContractError {
    ContractError::invalid("environment storage", "size overflow or invalid encoding")
}
