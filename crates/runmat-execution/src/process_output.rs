//! Bounded raw process streams, separate from numerical results and from
//! runtime-produced console receipts. EOF alone never certifies retirement.
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ContractError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessOutputReport {
    pub stdout: ProcessStreamOutput,
    pub stderr: ProcessStreamOutput,
}

impl ProcessOutputReport {
    pub fn validate(&self, maximum_stream_bytes: usize) -> Result<(), ContractError> {
        self.stdout.validate(maximum_stream_bytes)?;
        self.stderr.validate(maximum_stream_bytes)
    }

    /// Both original readers reached EOF, not just a producer completion claim.
    /// The process owner must independently establish physical retirement.
    pub fn drained_to_eof(&self) -> bool {
        self.stdout.completion == ProcessStreamCompletion::EndOfStream
            && self.stderr.completion == ProcessStreamCompletion::EndOfStream
    }
}

/// A retained prefix, not an assumption that all bytes were UTF-8. The whole
/// stream is drained even after this prefix fills. None denotes count overflow.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessStreamOutput {
    pub bytes: ProcessOutputBytes,
    pub observed_bytes: Option<u64>,
    pub truncated: bool,
    pub completion: ProcessStreamCompletion,
}

impl ProcessStreamOutput {
    pub fn validate(&self, maximum_stream_bytes: usize) -> Result<(), ContractError> {
        let retained = self.bytes.as_slice().len();
        if retained > maximum_stream_bytes
            || self
                .observed_bytes
                .is_some_and(|count| count < retained as u64)
            || (self.observed_bytes.is_none() && !self.truncated)
            || (self.completion != ProcessStreamCompletion::Transferred
                && self
                    .observed_bytes
                    .is_some_and(|count| (count > retained as u64) != self.truncated))
        {
            return Err(ContractError::invalid(
                "process output",
                "stream prefix exceeds its bound or contradicts its byte count",
            ));
        }
        Ok(())
    }
}

/// Immutable byte sharing across original evidence and report projections.
/// Cloning a report does not duplicate the retained prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessOutputBytes(Arc<Vec<u8>>);

impl ProcessOutputBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl Serialize for ProcessOutputBytes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProcessOutputBytes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<u8>::deserialize(deserializer).map(Self::new)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStreamCompletion {
    NotCaptured,
    Reading,
    EndOfStream,
    ReadFailed { kind: String },
    Transferred,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(observed_bytes: Option<u64>, truncated: bool) -> ProcessStreamOutput {
        ProcessStreamOutput {
            bytes: ProcessOutputBytes::new(vec![0, 255]),
            observed_bytes,
            truncated,
            completion: ProcessStreamCompletion::EndOfStream,
        }
    }

    #[test]
    fn stream_inventory_rejects_inconsistent_counts_and_bounds() {
        for (observed, truncated, maximum, valid) in [
            (Some(2), false, 2, true),
            (Some(3), true, 2, true),
            (None, true, 2, true),
            (Some(1), false, 2, false),
            (Some(2), true, 2, false),
            (Some(3), false, 2, false),
            (None, false, 2, false),
            (Some(2), false, 1, false),
        ] {
            assert_eq!(
                stream(observed, truncated).validate(maximum).is_ok(),
                valid,
                "observed={observed:?}, truncated={truncated}, maximum={maximum}",
            );
        }
    }

    #[test]
    fn report_shares_raw_bytes_and_round_trips_without_claiming_retirement() {
        let report = ProcessOutputReport {
            stdout: stream(Some(2), false),
            stderr: stream(Some(2), false),
        };
        let cloned = report.clone();
        assert!(Arc::ptr_eq(&report.stdout.bytes.0, &cloned.stdout.bytes.0));
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: ProcessOutputReport = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, report);
        assert!(decoded.drained_to_eof());
        let mut incomplete = decoded;
        incomplete.stderr.completion = ProcessStreamCompletion::Reading;
        assert!(!incomplete.drained_to_eof());
    }
}
