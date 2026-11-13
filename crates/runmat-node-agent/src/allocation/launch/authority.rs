//! Immutable allocation authority and role-specific child launch evidence.
use std::path::{Path, PathBuf};

use runmat_execution_transport_native::control::{NodeAllocation, WorkerBootstrapCredential};
use runmat_process_host::{clock::HostMonotonicReading, ProcessSpawnEvidence};

use super::super::Sandbox;
use super::WorkerJournalLaunch;

/// Node environment and allocation captured together for one child launch.
pub struct AllocationLaunchContext<'a> {
    pub runmat_executable: &'a Path,
    pub allocation: &'a NodeAllocation,
    pub sandbox: &'a Sandbox,
    pub server_url: &'a str,
    pub clock_domain_nonce: [u8; 16],
}

#[derive(Clone, Debug)]
pub struct DriverClockScope {
    pub run_id: String,
    pub allocation_lease_id: String,
    pub allocation_fence: u64,
    pub process_launch_nonce: [u8; 16],
    pub clock_domain_nonce: [u8; 16],
    pub request_directory: PathBuf,
    pub receipt_directory: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerRetirementScope {
    pub run_id: String,
    pub allocation_lease_id: String,
    pub allocation_fencing_token: u64,
    pub driver_fencing_token: u64,
    pub process_id: u32,
    pub process_launch_nonce: [u8; 16],
    pub clock_domain_nonce: [u8; 16],
    pub journal_directory: PathBuf,
    pub retired_monotonic: Option<HostMonotonicReading>,
    pub retired_at_unix_millis: u64,
}

/// A worker's bootstrap, physical-spawn evidence and journal are inseparable.
/// A driver instead owns its durable clock observer; invalid mixtures cannot
/// be passed to the process-launch boundary.
pub(super) enum ProcessLaunchAuthority<'a> {
    Worker {
        bootstrap: &'a WorkerBootstrapCredential,
        spawn_evidence: &'a ProcessSpawnEvidence,
        journal: WorkerJournalLaunch,
    },
    Driver {
        clock: DriverClockScope,
    },
}
