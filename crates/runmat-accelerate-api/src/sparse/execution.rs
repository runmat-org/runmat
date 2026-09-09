use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_execution::identity::{AttemptId, Digest};
use serde::{Deserialize, Serialize};

pub const SPARSE_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// A cheap, shareable cancellation flag for one numerical attempt.
///
/// This is deliberately data, not a user callback: providers may poll it at
/// their declared cooperative checkpoints without calling into runtime code.
#[derive(Clone, Debug, Default)]
pub struct SparseCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl SparseCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_shared(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseExecutionContext<'a> {
    pub attempt_id: AttemptId,
    pub operator_identity: Digest,
    pub provider_capability_schema_version: u16,
    pub provider_capability_revision: u64,
    pub cancellation: &'a SparseCancellationToken,
}

impl SparseExecutionContext<'_> {
    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.provider_capability_schema_version > 0,
            "sparse execution requires a positive provider capability schema version"
        );
        anyhow::ensure!(
            self.provider_capability_revision > 0,
            "sparse execution requires a positive provider capability revision"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseExecutionBinding {
    pub attempt_id: AttemptId,
    pub operator_identity: Digest,
    pub provider_device_id: u32,
    pub provider_capability_schema_version: u16,
    pub provider_capability_revision: u64,
    pub pattern_id: u64,
    pub matrix_id: u64,
    pub preconditioner_id: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseOperationTimings {
    pub preparation_ns: u64,
    pub execution_ns: u64,
    pub terminal_synchronization_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseOperationEvidence {
    pub schema_version: u16,
    pub binding: SparseExecutionBinding,
    pub timings: SparseOperationTimings,
    pub resident_pattern_bytes: u64,
    pub resident_matrix_bytes: u64,
    pub resident_preconditioner_bytes: u64,
    pub resident_solution_bytes: u64,
    pub host_upload_bytes: u64,
    pub host_download_bytes: u64,
    pub setup_dispatches: u64,
    pub spmv_dispatches: u64,
    pub preconditioner_dispatches: u64,
    pub vector_dispatches: u64,
    pub host_synchronizations: u32,
    pub device_readbacks: u32,
    pub peak_resident_bytes: u64,
    pub scratch_bytes: u64,
    pub cancellation_checks: u64,
    pub cancellation_observed: bool,
    pub used_host_fallback: bool,
}
