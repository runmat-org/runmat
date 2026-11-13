//! Ordered console evidence, separate from return values and lossy progress.
//! A decoded completion marker is a producer claim, not proof of execution.
use crate::ContractError;
use serde::{Deserialize, Serialize};

mod transport;
pub use transport::{ConsoleReceipt, ConsoleReceiver};
mod inventory;
pub use inventory::ConsoleInventory;
mod page;
pub use page::{ConsoleCursor, ConsoleFragment, ConsolePage, ConsoleReadBudget};
mod capture;
pub use capture::ConsoleCapture;
mod admission;
mod scopes;
mod budget;
pub use admission::{deserialize_controls, invocation_memory_bytes, ConsoleStorageProfile};
pub use budget::{
    deserialize_budget, ConsoleBudget, CONSOLE_RETAINED_COPY_FACTOR, CONSOLE_SCOPE_BYTES,
};
pub use scopes::{
    deserialize_scopes, deserialize_source_identity, original_root, validate_delegated_root,
    validate_scopes, validate_source_controls, ConsoleSourceIdentity, ScopedConsoleTranscript,
    ScopedConsoleTranscriptRef, MAX_CONSOLE_SCOPES,
};

pub const CONSOLE_TRANSCRIPT_SCHEMA_VERSION: u16 = 2;
pub const MAX_CONSOLE_RECORDS: u32 = 65_536;
pub const MAX_CONSOLE_TEXT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleStream {
    Stdout,
    Stderr,
    ClearScreen,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleLimits {
    pub max_records: u32,
    pub max_text_bytes: u64,
}

impl ConsoleLimits {
    pub fn validate(self) -> Result<(), ContractError> {
        if self.max_records == 0
            || self.max_records > MAX_CONSOLE_RECORDS
            || self.max_text_bytes == 0
            || self.max_text_bytes > MAX_CONSOLE_TEXT_BYTES
        {
            return Err(ContractError::invalid(
                "console limits",
                "outside supported bounds",
            ));
        }
        Ok(())
    }

    /// Requested record slots plus UTF-8 payload storage. Excludes allocator
    /// overhead, incoming formatting buffers, copies and transport encoding.
    pub fn capture_storage_bytes(self) -> Result<u64, ContractError> {
        self.validate()?;
        Ok(
            u64::from(self.max_records) * std::mem::size_of::<ConsoleRecord>() as u64
                + self.max_text_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleRejection {
    RecordLimit,
    TextByteLimit,
    InvalidControlRecord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ConsoleCompletion {
    Complete,
    Interrupted,
    Rejected { reason: ConsoleRejection },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleRecord {
    pub sequence: u64,
    pub stream: ConsoleStream,
    pub text: Box<str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleTranscript {
    schema_version: u16,
    limits: ConsoleLimits,
    #[serde(deserialize_with = "deserialize_records")]
    records: Vec<ConsoleRecord>,
    text_bytes: u64,
    completion: ConsoleCompletion,
    #[serde(deserialize_with = "deserialize_source_receipt")]
    source_receipt: Option<Box<ConsoleReceipt>>,
}

impl ConsoleTranscript {
    pub fn source_receipt(&self) -> Option<&ConsoleReceipt> {
        self.source_receipt.as_deref()
    }
    pub fn records(&self) -> &[ConsoleRecord] {
        &self.records
    }
    pub fn limits(&self) -> ConsoleLimits {
        self.limits
    }
    pub fn text_bytes(&self) -> u64 {
        self.text_bytes
    }
    pub fn completion(&self) -> ConsoleCompletion {
        self.completion
    }

    /// Validate the retained inventory. Does not establish source authenticity
    /// or turn an interrupted prefix into complete output.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != CONSOLE_TRANSCRIPT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: CONSOLE_TRANSCRIPT_SCHEMA_VERSION,
            });
        }
        self.limits.validate()?;
        if self.records.len() > self.limits.max_records as usize {
            return Err(ContractError::invalid(
                "console transcript",
                "record bound exceeded",
            ));
        }
        let mut bytes = 0_u64;
        for (index, record) in self.records.iter().enumerate() {
            if record.sequence != index as u64 + 1
                || (record.stream == ConsoleStream::ClearScreen && !record.text.is_empty())
            {
                return Err(ContractError::invalid(
                    "console transcript",
                    "invalid ordered record",
                ));
            }
            bytes = bytes.checked_add(record.text.len() as u64).ok_or_else(|| {
                ContractError::invalid("console transcript", "byte count overflow")
            })?;
        }
        if bytes != self.text_bytes || bytes > self.limits.max_text_bytes {
            return Err(ContractError::invalid(
                "console transcript",
                "invalid UTF-8 byte inventory",
            ));
        }
        if let Some(receipt) = &self.source_receipt {
            receipt.validate_inventory(self)?;
        }
        Ok(())
    }
}

fn deserialize_source_receipt<'de, D>(
    deserializer: D,
) -> Result<Option<Box<ConsoleReceipt>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Deserialize::deserialize(deserializer)
}

fn deserialize_records<'de, D>(deserializer: D) -> Result<Vec<ConsoleRecord>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let records = Vec::<ConsoleRecord>::deserialize(deserializer)?;
    Ok(records.into_boxed_slice().into_vec())
}

/// Required optional evidence: omitted fields cannot imply a complete empty console.
pub fn deserialize_transcript<'de, D>(
    deserializer: D,
) -> Result<Option<Box<ConsoleTranscript>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let transcript: Option<Box<ConsoleTranscript>> = Deserialize::deserialize(deserializer)?;
    if let Some(transcript) = &transcript {
        transcript.validate().map_err(serde::de::Error::custom)?;
    }
    Ok(transcript)
}

#[cfg(test)]
mod tests;
