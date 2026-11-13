use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use chrono::Utc;
use runmat_execution::security::{
    EndpointTrustPolicy, WorkerTreeRetirementEvidenceV1, MAX_RETIREMENT_WIRE_BYTES,
};

use crate::allocation::WorkerRetirementScope;
use crate::enrollment::NodeCredential;
use crate::{AgentError, AgentResult};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "camelCase", deny_unknown_fields)]
pub(super) enum PendingRetirementRecord {
    /// Original process and clock ownership was durably retained before any
    /// destructive process-tree operation. This state is intentionally not
    /// promoted after restart: the old live clock domain cannot be recreated.
    Retiring {
        scope: Option<WorkerRetirementScope>,
    },
    /// The process tree retired and the exact signed evidence (for workers)
    /// is durable and independently retryable.
    Certified {
        evidence: Option<WorkerTreeRetirementEvidenceV1>,
    },
}

pub(super) fn load(
    state_directory: &Path,
    credential: &NodeCredential,
    maximum_allocations: usize,
) -> AgentResult<BTreeMap<String, PendingRetirementRecord>> {
    use std::io::Read as _;

    let path = state_directory.join("pending-worker-retirements.json");
    let limit = budget(maximum_allocations)?;
    let read = std::fs::File::open(path).and_then(|file| {
        let mut bytes = Vec::with_capacity(limit.min(16 * 1024));
        file.take((limit + 1) as u64).read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    match read {
        Ok(bytes) => {
            if bytes.len() > limit {
                return Err(AgentError::AllocationRejected(
                    "retirement journal exceeds bounded capacity".into(),
                ));
            }
            let pending: BTreeMap<String, PendingRetirementRecord> =
                serde_json::from_slice(&bytes)?;
            validate(&pending, credential)?;
            Ok(pending)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(AgentError::Io(error)),
    }
}

fn validate(
    pending: &BTreeMap<String, PendingRetirementRecord>,
    credential: &NodeCredential,
) -> AgentResult<()> {
    let policy = EndpointTrustPolicy {
        permitted_tiers: BTreeSet::new(),
        trusted_identity_fingerprints: BTreeSet::from([credential.identity_fingerprint.clone()]),
        allowed_attestation_classes: BTreeSet::new(),
        require_pinned_identity: true,
        now_unix_millis: Utc::now().timestamp_millis().try_into().unwrap_or(0),
        maximum_clock_skew_millis: 30_000,
    };
    for (allocation_id, record) in pending {
        if let PendingRetirementRecord::Retiring { scope: Some(scope) } = record {
            if scope.allocation_lease_id != *allocation_id
                || scope.retired_monotonic.is_some()
                || scope.retired_at_unix_millis != 0
            {
                return Err(AgentError::AllocationRejected(
                    "retiring process journal scope mismatch".into(),
                ));
            }
        }
        if let PendingRetirementRecord::Certified {
            evidence: Some(evidence),
        } = record
        {
            if evidence.allocation_lease_id != *allocation_id
                || evidence.node_id != credential.node_id
                || evidence.cluster_id != credential.cluster_id
                || evidence.org_id != credential.org_id
            {
                return Err(AgentError::AllocationRejected(
                    "retirement journal scope mismatch".into(),
                ));
            }
            policy
                .verify_worker_tree_retirement(evidence, &credential.identity_fingerprint)
                .map_err(|error| AgentError::AllocationRejected(error.to_string()))?;
        }
    }
    Ok(())
}

pub(super) fn prepare(
    state_directory: &Path,
    maximum_allocations: usize,
    pending: &mut BTreeMap<String, PendingRetirementRecord>,
    allocation_id: &str,
    scope: Option<&WorkerRetirementScope>,
) -> AgentResult<()> {
    if pending.contains_key(allocation_id) {
        return Ok(());
    }
    let mut next = pending.clone();
    next.insert(
        allocation_id.to_string(),
        PendingRetirementRecord::Retiring {
            scope: scope.cloned(),
        },
    );
    persist(state_directory, maximum_allocations, &next)?;
    *pending = next;
    Ok(())
}

pub(super) fn persist(
    state_directory: &Path,
    maximum_allocations: usize,
    pending: &BTreeMap<String, PendingRetirementRecord>,
) -> AgentResult<()> {
    let path = state_directory.join("pending-worker-retirements.json");
    let temporary = state_directory.join(".pending-worker-retirements.pending");
    let bytes = serde_json::to_vec(pending)?;
    if bytes.len() > budget(maximum_allocations)? {
        return Err(AgentError::AllocationRejected(
            "pending retirement journal exceeds retained allocation budget".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt as _;
        validate_staging_slot(&temporary)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        use std::io::Write as _;
        validate_staging_slot(&temporary)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    runmat_process_host::durable_file::publish_synced_file(&temporary, &path)?;
    Ok(())
}

fn validate_staging_slot(path: &Path) -> AgentResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(AgentError::AllocationRejected(
            "pending retirement staging slot is not a regular file".into(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn budget(maximum_allocations: usize) -> AgentResult<usize> {
    maximum_allocations
        .checked_mul(MAX_RETIREMENT_WIRE_BYTES + 512)
        .and_then(|value| value.checked_add(1024))
        .ok_or_else(|| {
            AgentError::AllocationRejected("pending retirement allocation budget overflowed".into())
        })
}
