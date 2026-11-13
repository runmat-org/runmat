use serde::{Deserialize, Serialize};

/// Scoped JSON metadata is at most 684 bytes with maximal controls, all four
/// typed identities, the longest completion tag and an original source receipt.
/// Records/text have separate quotas; round this metadata bound to 1024 bytes.
pub const CONSOLE_SCOPE_BYTES: u64 = 1024;
/// Original retirement/terminal evidence, one single-flight checkpoint, and
/// bounded scheduler transition copies. Encoding work is quoted separately.
pub const CONSOLE_RETAINED_COPY_FACTOR: u64 = 8;
const MAX_CONSOLE_SCOPES_U32: u32 = super::MAX_CONSOLE_SCOPES as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleBudget {
    pub max_scopes: u32,
    pub max_records: u64,
    pub max_text_bytes: u64,
    pub max_encoded_bytes: u64,
    pub max_retained_bytes: u64,
}

impl ConsoleBudget {
    /// Retained component plus bounded plain/encrypted/framed encoding work.
    /// Formatting, allocator overhead and domain execution remain separate.
    pub fn minimum_memory_bytes(self) -> Result<u64, crate::ContractError> {
        self.validate()?;
        self.max_encoded_bytes
            .checked_mul(3)
            .and_then(|encoded| self.max_retained_bytes.checked_add(encoded))
            .ok_or_else(|| crate::ContractError::invalid("console budget", "storage overflow"))
    }
    /// Finite lifetime component quotas from the authenticated driver memory.
    /// The eight-copy component allows existing scheduler/checkpoint transitions;
    /// these quotas are not an assertion about total execution RSS.
    pub fn for_driver_allocation(memory_bytes: u64) -> Self {
        let retained = (memory_bytes / 8).min(64 * 1024 * 1024);
        let encoded = (memory_bytes / 4).min(64 * 1024 * 1024);
        Self {
            max_scopes: (retained / CONSOLE_SCOPE_BYTES).min(u64::from(MAX_CONSOLE_SCOPES_U32))
                as u32,
            max_records: (retained
                / (std::mem::size_of::<super::ConsoleRecord>() as u64
                    * CONSOLE_RETAINED_COPY_FACTOR))
                .min(u64::from(super::MAX_CONSOLE_RECORDS)),
            max_text_bytes: (retained / CONSOLE_RETAINED_COPY_FACTOR)
                .min(super::MAX_CONSOLE_TEXT_BYTES),
            max_encoded_bytes: encoded,
            max_retained_bytes: retained,
        }
    }

    pub fn single(limits: super::ConsoleLimits) -> Result<Self, crate::ContractError> {
        let profile = limits.storage_profile()?;
        Ok(Self {
            max_scopes: 1,
            max_records: u64::from(limits.max_records),
            max_text_bytes: limits.max_text_bytes,
            max_encoded_bytes: profile.max_encoded_bytes + CONSOLE_SCOPE_BYTES,
            max_retained_bytes: (profile.capture_bytes + CONSOLE_SCOPE_BYTES)
                * CONSOLE_RETAINED_COPY_FACTOR,
        })
    }

    pub fn validate(self) -> Result<(), crate::ContractError> {
        let global = super::ConsoleLimits {
            max_records: super::MAX_CONSOLE_RECORDS,
            max_text_bytes: super::MAX_CONSOLE_TEXT_BYTES,
        }
        .storage_profile()?;
        let metadata = u64::from(MAX_CONSOLE_SCOPES_U32) * CONSOLE_SCOPE_BYTES;
        if self.max_scopes == 0
            || self.max_scopes > MAX_CONSOLE_SCOPES_U32
            || self.max_records == 0
            || self.max_records > u64::from(super::MAX_CONSOLE_RECORDS)
            || self.max_text_bytes == 0
            || self.max_text_bytes > super::MAX_CONSOLE_TEXT_BYTES
            || self.max_encoded_bytes == 0
            || self.max_encoded_bytes > global.max_encoded_bytes + metadata
            || self.max_retained_bytes == 0
            || self.max_retained_bytes
                > (global.capture_bytes + metadata) * CONSOLE_RETAINED_COPY_FACTOR
        {
            return Err(crate::ContractError::invalid(
                "console budget",
                "terminal aggregate exceeds supported bounds",
            ));
        }
        Ok(())
    }
}

pub fn deserialize_budget<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ConsoleBudget>, D::Error> {
    let budget = Option::<ConsoleBudget>::deserialize(deserializer)?;
    if let Some(budget) = budget {
        budget.validate().map_err(serde::de::Error::custom)?;
    }
    Ok(budget)
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::console::{
        ConsoleCapture, ConsoleCompletion, ConsoleLimits, ConsoleReceiver, ConsoleRejection,
        ScopedConsoleTranscriptRef,
    };

    #[test]
    fn scoped_metadata_including_source_receipt_fits_its_canonical_quote() {
        let limits = ConsoleLimits {
            max_records: super::super::MAX_CONSOLE_RECORDS,
            max_text_bytes: super::super::MAX_CONSOLE_TEXT_BYTES,
        };
        let completions = [
            ConsoleCompletion::Complete,
            ConsoleCompletion::Interrupted,
            ConsoleCompletion::Rejected {
                reason: ConsoleRejection::RecordLimit,
            },
            ConsoleCompletion::Rejected {
                reason: ConsoleRejection::TextByteLimit,
            },
            ConsoleCompletion::Rejected {
                reason: ConsoleRejection::InvalidControlRecord,
            },
        ];
        let mut sizes = Vec::new();
        for completion in completions {
            for with_receipt in [false, true] {
                let mut transcript = ConsoleCapture::new(limits).unwrap().finish(true);
                transcript.completion = completion;
                if with_receipt {
                    let receipt = transcript.receipt().unwrap();
                    transcript = ConsoleReceiver::new(limits)
                        .unwrap()
                        .finish(Some(&receipt))
                        .unwrap();
                }
                transcript.validate().unwrap();
                let scoped = ScopedConsoleTranscriptRef {
                    scope_id: crate::ExecutionScopeId::from_bytes([0xff; 16]),
                    pool_id: crate::PoolId::from_bytes([0xff; 16]),
                    task_id: crate::TaskId::from_bytes([0xff; 16]),
                    attempt_id: crate::identity::AttemptId::from_bytes([0xff; 16]),
                    transcript: &transcript,
                };
                sizes.push(serde_json::to_vec(&scoped).unwrap().len() as u64);
            }
        }
        assert_eq!(sizes, [385, 597, 388, 603, 409, 645, 412, 651, 419, 665]);
        // Empty inventories have zero in three metadata counters. Supported
        // nonempty inventories add only these decimal widths; the record array
        // itself (including record separators) is covered by record/text quotas.
        let count_growth = limits.max_records.to_string().len() as u64 - 1;
        let byte_growth = limits.max_text_bytes.to_string().len() as u64 - 1;
        let largest_metadata = sizes.into_iter().max().unwrap() + count_growth + 2 * byte_growth;
        assert_eq!(largest_metadata, 683);
        assert_eq!(largest_metadata + 1, 684); // outer scope-array separator
        assert!(largest_metadata < CONSOLE_SCOPE_BYTES);
    }
}
