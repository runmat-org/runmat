//! Streaming inventory for a publisher that retains original records elsewhere.
use super::*;
use crate::Digest;
use sha2::{Digest as _, Sha256};

#[derive(Clone)]
pub struct ConsoleInventory {
    limits: ConsoleLimits,
    records: u64,
    bytes: u64,
    digest: RecordDigest,
}

impl ConsoleInventory {
    pub fn new(limits: ConsoleLimits) -> Result<Self, ContractError> {
        limits.validate()?;
        Ok(Self {
            limits,
            records: 0,
            bytes: 0,
            digest: RecordDigest::new(),
        })
    }

    /// Reject before changing counters or the digest; retains no record text.
    pub fn push(&mut self, record: &ConsoleRecord) -> Result<(), ContractError> {
        let bytes = self
            .bytes
            .checked_add(record.text.len() as u64)
            .filter(|n| *n <= self.limits.max_text_bytes);
        if record.sequence != self.records + 1
            || self.records >= u64::from(self.limits.max_records)
            || bytes.is_none()
            || (record.stream == ConsoleStream::ClearScreen && !record.text.is_empty())
        {
            return Err(ContractError::invalid(
                "console inventory",
                "invalid next original record",
            ));
        }
        self.bytes = bytes.expect("validated byte inventory");
        self.records += 1;
        self.digest.push(record);
        Ok(())
    }

    pub fn receipt(&self, completion: ConsoleCompletion) -> ConsoleReceipt {
        ConsoleReceipt {
            limits: self.limits,
            record_count: self.records,
            text_bytes: self.bytes,
            completion,
            records_digest: self.digest.finish(),
        }
    }
}

#[derive(Clone)]
pub(super) struct RecordDigest(Sha256);
impl RecordDigest {
    pub(super) fn new() -> Self {
        let mut hash = Sha256::new();
        hash.update(b"runmat-console-records-v1\0");
        Self(hash)
    }
    pub(super) fn push(&mut self, record: &ConsoleRecord) {
        self.0.update(record.sequence.to_be_bytes());
        self.0.update([match record.stream {
            ConsoleStream::Stdout => 0,
            ConsoleStream::Stderr => 1,
            ConsoleStream::ClearScreen => 2,
        }]);
        self.0.update((record.text.len() as u64).to_be_bytes());
        self.0.update(record.text.as_bytes());
    }
    pub(super) fn finish(&self) -> Digest {
        Digest::from_bytes(self.0.clone().finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streamed_original_identity_matches_independent_wire_and_rejection_is_atomic() {
        let limits = ConsoleLimits {
            max_records: 2,
            max_text_bytes: 4,
        };
        let original = ConsoleRecord {
            sequence: 1,
            stream: ConsoleStream::Stdout,
            text: "λ".into(),
        };
        let mut inventory = ConsoleInventory::new(limits).unwrap();
        inventory.push(&original).unwrap();
        let receipt = inventory.receipt(ConsoleCompletion::Interrupted);
        let mut bytes = b"runmat-console-records-v1\0".to_vec();
        bytes.extend(1_u64.to_be_bytes());
        bytes.push(0);
        bytes.extend(2_u64.to_be_bytes());
        bytes.extend("λ".as_bytes());
        assert_eq!(receipt.records_digest, Digest::sha256(bytes));
        for altered in [
            original,
            ConsoleRecord {
                sequence: 2,
                stream: ConsoleStream::ClearScreen,
                text: "bad".into(),
            },
            ConsoleRecord {
                sequence: 2,
                stream: ConsoleStream::Stdout,
                text: "large".into(),
            },
        ] {
            assert!(inventory.push(&altered).is_err());
            assert_eq!(inventory.receipt(ConsoleCompletion::Interrupted), receipt);
        }
    }
}
