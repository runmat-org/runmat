//! Fsynced per-attempt admission and result inventory in the node allocation sandbox.
//!
//! One physical worker allocation owns at most 256 encrypted regular entries
//! of at most 4 KiB each, plus one 4 KiB atomic-publication slot and directory metadata. At the cap, admission is
//! deferred before execution; only a server-acknowledged durable checkpoint
//! may reclaim terminal entries. An ACK of a physical receipt alone cannot
//! reclaim them because a prior checkpoint may still be restored.

use rand::RngCore as _;
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::security::{
    RetiredResultInventoryV1, WorkerAttemptJournalEntryV1, MAX_RETIREMENT_ATTEMPTS,
    MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES, MAX_RETIREMENT_JOURNAL_ENTRY_BYTES,
    RETIREMENT_JOURNAL_STAGING_FILE,
};
use runmat_execution_runner::{AttemptReport, AttemptRequest};
use runmat_process_host::clock::observe_monotonic;

use super::CheckpointedAttempt;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct WorkerAttemptJournalOwner {
    pub(crate) directory: PathBuf,
    pub(crate) run_id: String,
    pub(crate) allocation_lease_id: String,
    pub(crate) allocation_fencing_token: u64,
    pub(crate) driver_fencing_token: u64,
    pub(crate) process_launch_nonce: [u8; 16],
    pub(crate) clock_domain_nonce: [u8; 16],
}

pub(super) struct WorkerAttemptJournal {
    owner: WorkerAttemptJournalOwner,
    worker_id: WorkerId,
    run_key: RunKeyMaterial,
    entries: BTreeMap<AttemptId, WorkerAttemptJournalEntryV1>,
}

impl WorkerAttemptJournal {
    pub(super) fn is_full(&self) -> bool {
        self.entries.len() >= MAX_RETIREMENT_ATTEMPTS
    }

    pub(super) fn new(
        owner: WorkerAttemptJournalOwner,
        worker_id: WorkerId,
        run_key: RunKeyMaterial,
    ) -> NativeExecutionResult<Self> {
        if !owner.directory.is_absolute()
            || !owner.directory.is_dir()
            || owner.run_id.is_empty()
            || owner.allocation_lease_id.is_empty()
            || owner.allocation_fencing_token == 0
            || owner.driver_fencing_token == 0
            || owner.process_launch_nonce.iter().all(|value| *value == 0)
            || owner.clock_domain_nonce.iter().all(|value| *value == 0)
            || worker_id
                != WorkerId::derive(&[
                    owner.run_id.as_bytes(),
                    owner.allocation_lease_id.as_bytes(),
                ])
            || std::fs::read_dir(&owner.directory)
                .map_err(protocol)?
                .next()
                .is_some()
        {
            return Err(protocol(
                "worker attempt journal is not a fresh owned directory",
            ));
        }
        Ok(Self {
            owner,
            worker_id,
            run_key,
            entries: BTreeMap::new(),
        })
    }

    pub(super) fn begin(&mut self, request: &AttemptRequest) -> NativeExecutionResult<()> {
        if request.worker_id != self.worker_id
            || request.driver_fence != self.owner.driver_fencing_token
            || self.entries.contains_key(&request.id)
            || self.entries.len() >= MAX_RETIREMENT_ATTEMPTS
        {
            return Err(protocol(
                "worker attempt journal lacks an exact admitted slot",
            ));
        }
        let started = observe_monotonic().map_err(protocol)?;
        let entry = WorkerAttemptJournalEntryV1 {
            schema_version: 1,
            run_id: self.owner.run_id.clone(),
            allocation_lease_id: self.owner.allocation_lease_id.clone(),
            allocation_fencing_token: self.owner.allocation_fencing_token,
            driver_fencing_token: self.owner.driver_fencing_token,
            process_launch_nonce: self.owner.process_launch_nonce,
            attempt_id: request.id,
            task_id: request.task_id,
            worker_id: request.worker_id,
            ordinal: request.ordinal,
            request_digest: runmat_execution::Digest::sha256_serialized(request)
                .map_err(protocol)?,
            clock_domain_nonce: self.owner.clock_domain_nonce,
            started_monotonic_ticks: started.ticks,
            ticks_per_second: started.ticks_per_second,
            finished_monotonic_ticks: None,
            published_result_inventory: None,
        };
        self.write(&entry)?;
        self.entries.insert(request.id, entry);
        Ok(())
    }

    pub(super) fn complete(
        &mut self,
        attempt_id: AttemptId,
        report: &AttemptReport,
    ) -> NativeExecutionResult<()> {
        let previous = self
            .entries
            .get(&attempt_id)
            .ok_or_else(|| protocol("completed attempt has no journal admission"))?;
        if previous.finished_monotonic_ticks.is_some() {
            return Err(protocol("attempt result inventory was already committed"));
        }
        let end = observe_monotonic().map_err(protocol)?;
        if end.ticks_per_second != previous.ticks_per_second
            || end.ticks <= previous.started_monotonic_ticks
        {
            return Err(protocol("attempt crossed an unverified clock domain"));
        }
        let published_result_inventory = match report {
            AttemptReport::Succeeded { result } => {
                match result.resources() {
                    Some(measured) => Some(RetiredResultInventoryV1 {
                        artifact_bytes: measured.resources.artifact_bytes,
                        egress_bytes: measured.resources.usage.egress_bytes,
                        relay_bytes: measured.resources.usage.relay_bytes,
                    }),
                    None if matches!(
                        result,
                        runmat_execution_runner::AttemptSuccess::Spmd { .. }
                    ) =>
                    {
                        // Spmd publishes no `result_objects` inventory by
                        // AttemptSuccess contract; this is a declared-output
                        // count, not a network-traffic measurement.
                        Some(RetiredResultInventoryV1 {
                            artifact_bytes: 0,
                            egress_bytes: 0,
                            relay_bytes: 0,
                        })
                    }
                    None => return Err(protocol("successful result has no exact accounting")),
                }
            }
            _ => None,
        };
        let mut entry = previous.clone();
        entry.finished_monotonic_ticks = Some(end.ticks);
        entry.published_result_inventory = published_result_inventory;
        self.write(&entry)?;
        self.entries.insert(attempt_id, entry);
        Ok(())
    }

    pub(super) fn prune_after_checkpoint(
        &mut self,
        sequence: u64,
        attempts: &[CheckpointedAttempt],
    ) -> NativeExecutionResult<()> {
        if sequence == 0 || attempts.len() > MAX_RETIREMENT_ATTEMPTS {
            return Err(protocol(
                "durable checkpoint prune exceeds its authenticated scope",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut removable = Vec::new();
        for attempt in attempts {
            if !seen.insert(attempt.attempt_id) {
                return Err(protocol("durable checkpoint prune repeats an attempt"));
            }
            if let Some(entry) = self.entries.get(&attempt.attempt_id) {
                if entry.request_digest != attempt.request_digest
                    || entry.finished_monotonic_ticks.is_none()
                {
                    return Err(protocol(
                        "durable checkpoint differs from terminal attempt journal",
                    ));
                }
                removable.push(attempt.attempt_id);
            }
        }
        for attempt_id in &removable {
            let path = entry_path(&self.owner.directory, *attempt_id);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(protocol(error)),
            }
        }
        sync_directory(&self.owner.directory)?;
        for attempt_id in removable {
            self.entries.remove(&attempt_id);
        }
        Ok(())
    }

    fn write(&self, entry: &WorkerAttemptJournalEntryV1) -> NativeExecutionResult<()> {
        entry.validate_shape().map_err(protocol)?;
        let journal = serde_json::to_vec(entry).map_err(protocol)?;
        if journal.len() > MAX_RETIREMENT_JOURNAL_ENTRY_BYTES {
            return Err(protocol("attempt journal entry exceeds its physical grant"));
        }
        // The clear run-object context includes a plaintext digest. An
        // independent secret inside the plaintext makes that digest resistant
        // to guessing even when coarse owner nonces are public.
        let mut plaintext = Vec::with_capacity(32 + journal.len());
        let mut secret = [0_u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut secret);
        plaintext.extend_from_slice(&secret);
        plaintext.extend_from_slice(&journal);
        let bytes = super::crypto::seal_object(
            &self.run_key,
            &self.owner.run_id,
            EncryptionPurpose::RetirementReceipt,
            &plaintext,
        )?;
        if bytes.len() > MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES {
            return Err(protocol(
                "encrypted attempt journal exceeds its physical grant",
            ));
        }
        let destination = entry_path(&self.owner.directory, entry.attempt_id);
        // Journal mutation is serialized by the worker's journal mutex. A
        // fixed staging slot bounds crash debris and is never an admission.
        let temporary = self.owner.directory.join(RETIREMENT_JOURNAL_STAGING_FILE);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary).map_err(protocol)?;
        file.write_all(&bytes).map_err(protocol)?;
        file.sync_all().map_err(protocol)?;
        runmat_process_host::durable_file::publish_synced_file(&temporary, &destination)
            .map_err(protocol)
    }
}

fn entry_path(directory: &Path, attempt_id: AttemptId) -> PathBuf {
    directory.join(format!("{attempt_id}.receipt"))
}

fn sync_directory(directory: &Path) -> NativeExecutionResult<()> {
    #[cfg(unix)]
    std::fs::File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(protocol)?;
    #[cfg(windows)]
    let _ = directory;
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
