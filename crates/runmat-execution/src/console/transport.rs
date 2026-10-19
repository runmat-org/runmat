//! Ordered source transport. Receipts establish consistency, not execution authenticity.
use super::*;
use crate::Digest;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleReceipt {
    pub limits: ConsoleLimits,
    pub record_count: u64,
    pub text_bytes: u64,
    pub completion: ConsoleCompletion,
    pub records_digest: Digest,
}

impl ConsoleReceipt {
    pub(super) fn validate_inventory(
        &self,
        transcript: &ConsoleTranscript,
    ) -> Result<(), ContractError> {
        if self.limits != transcript.limits
            || self.record_count != transcript.records.len() as u64
            || self.text_bytes != transcript.text_bytes
            || self.completion != transcript.completion
            || self.records_digest != records_digest(&transcript.records)
        {
            return Err(ContractError::invalid(
                "console receipt",
                "retained source claim differs from inventory",
            ));
        }
        Ok(())
    }
}

impl ConsoleTranscript {
    pub fn receipt(&self) -> Result<ConsoleReceipt, ContractError> {
        self.validate()?;
        Ok(ConsoleReceipt {
            limits: self.limits,
            record_count: self.records.len() as u64,
            text_bytes: self.text_bytes,
            completion: self.completion,
            records_digest: records_digest(&self.records),
        })
    }
}

/// One receiver for one original source's admitted controls. Never sorts, drops,
/// or repairs records; a missing terminal receipt leaves the retained prefix interrupted.
pub struct ConsoleReceiver {
    capture: ConsoleCapture,
}

impl ConsoleReceiver {
    pub fn records(&self) -> &[ConsoleRecord] {
        self.capture.records()
    }
    pub fn new(limits: ConsoleLimits) -> Result<Self, ContractError> {
        Ok(Self {
            capture: ConsoleCapture::new(limits)?,
        })
    }
    pub fn push(&mut self, record: ConsoleRecord) -> Result<(), ContractError> {
        if record.sequence != self.capture.records().len() as u64 + 1 {
            return Err(ContractError::invalid(
                "console transport",
                "record sequence is not contiguous",
            ));
        }
        self.capture.push(record.stream, record.text.into_string())
    }
    pub fn verify_receipt(&self, receipt: &ConsoleReceipt) -> Result<(), ContractError> {
        if self.capture.limits() != receipt.limits
            || self.capture.records().len() as u64 != receipt.record_count
            || self.capture.text_bytes() != receipt.text_bytes
            || records_digest(self.capture.records()) != receipt.records_digest
        {
            return Err(ContractError::invalid(
                "console transport",
                "terminal receipt differs from original prefix",
            ));
        }
        Ok(())
    }
    pub fn finish(
        self,
        receipt: Option<&ConsoleReceipt>,
    ) -> Result<ConsoleTranscript, ContractError> {
        if let Some(receipt) = receipt {
            self.verify_receipt(receipt)?;
        }
        let mut transcript = self.capture.finish(false);
        if let Some(receipt) = receipt {
            transcript.completion = receipt.completion;
            transcript.source_receipt = Some(Box::new(receipt.clone()));
        }
        transcript.validate()?;
        Ok(transcript)
    }
}

fn records_digest(records: &[ConsoleRecord]) -> Digest {
    let mut hash = super::inventory::RecordDigest::new();
    for record in records {
        hash.push(record);
    }
    hash.finish()
}

#[cfg(test)]
mod tests;
