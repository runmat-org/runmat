//! One original provider publishes immutable writes and a small atomic prefix head.
//! The existing filesystem owns publication; the existing lease owns the writer.
use super::filesystem::{atomic_json, immutable_json, read_optional_json_bounded};
use crate::attempt_scratch::CONSOLE_HEAD_BYTES;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{console::*, Digest};
use runmat_execution_runner::AttemptRequest;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

mod read;
pub(super) use read::{read_transcript, ConsoleReadPlan};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema_version: u16,
    request_digest: Digest,
    limits: ConsoleLimits,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    inventory: ConsoleReceipt,
    last_record: Option<Digest>,
    source_receipt: Option<ConsoleReceipt>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredRecord<R = ConsoleRecord> {
    previous: Option<Digest>,
    record: R,
}

pub(super) struct ConsoleJournal {
    root: PathBuf,
    head: Head,
    inventory: ConsoleInventory,
}

pub(super) fn validate_reservation(
    task: &runmat_execution::task::TaskRequest,
) -> NativeExecutionResult<()> {
    crate::attempt_scratch::remaining_bytes(task.resources.scratch_bytes, 0, task.console)?;
    Ok(())
}

impl ConsoleJournal {
    pub(super) fn create(
        attempt: &super::model::DurableProgramAttempt,
        lease: &super::DurableAttemptLease,
    ) -> NativeExecutionResult<Option<Self>> {
        lease.validate(&attempt.job_directory, &attempt.request)?;
        validate_reservation(&attempt.request.task)?;
        let Some(limits) = attempt.request.task.console else {
            return Ok(None);
        };
        let root = directory(&attempt.job_directory, &attempt.request);
        let header = header(&attempt.request)?;
        let inventory = ConsoleInventory::new(limits).map_err(invalid)?;
        let head = Head {
            inventory: inventory.receipt(ConsoleCompletion::Interrupted),
            last_record: None,
            source_receipt: None,
        };
        // Publish the head before the original assignment marker. Readers discover
        // the marker first, so initialization cannot expose a headless journal.
        // Reopening a nonempty journal cannot erase or replay the original source.
        immutable_json(&root.join("head.json"), &head)?;
        immutable_json(&root.join("original.json"), &header)?;
        Ok(Some(Self {
            root,
            head,
            inventory,
        }))
    }

    pub(super) fn push(&mut self, record: &ConsoleRecord) -> NativeExecutionResult<()> {
        if self.head.source_receipt.is_some() {
            return Err(invalid("console journal is sealed"));
        }
        let mut inventory = self.inventory.clone();
        inventory.push(record).map_err(invalid)?;
        let stored = StoredRecord {
            previous: self.head.last_record,
            record,
        };
        let digest = json_digest(&stored)?;
        immutable_json(&record_path(&self.root, record.sequence), &stored)?;
        let head = Head {
            inventory: inventory.receipt(ConsoleCompletion::Interrupted),
            last_record: Some(digest),
            source_receipt: None,
        };
        // A record is observable only after its retained data and head both publish.
        atomic_json(&self.root.join("head.json"), &head)?;
        self.head = head;
        self.inventory = inventory;
        Ok(())
    }

    pub(super) fn finish(&mut self, receipt: &ConsoleReceipt) -> NativeExecutionResult<()> {
        if self.head.source_receipt.is_some()
            || self.inventory.receipt(receipt.completion) != *receipt
        {
            return Err(invalid(
                "console footer differs from original published inventory",
            ));
        }
        let mut head = self.head.clone();
        head.inventory = receipt.clone();
        head.source_receipt = Some(receipt.clone());
        atomic_json(&self.root.join("head.json"), &head)?;
        self.head = head;
        Ok(())
    }
}

fn header(request: &AttemptRequest) -> NativeExecutionResult<Header> {
    Ok(Header {
        schema_version: 1,
        request_digest: json_digest(request)?,
        limits: request
            .task
            .console
            .ok_or_else(|| invalid("original attempt did not request console"))?,
    })
}
fn directory(job: &Path, request: &AttemptRequest) -> PathBuf {
    job.join(format!("{}.console", request.id))
}
fn record_path(root: &Path, sequence: u64) -> PathBuf {
    root.join(format!("{sequence}.json"))
}

// A streamed native metadata identity; not a second portable request authority.
fn json_digest(value: &impl Serialize) -> NativeExecutionResult<Digest> {
    crate::metadata_identity::json_digest(
        b"runmat-native-console-metadata-v1\0",
        value,
        u64::from(super::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
    )
}
fn invalid(message: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.to_string())
}

#[cfg(all(test, unix))]
mod tests;
