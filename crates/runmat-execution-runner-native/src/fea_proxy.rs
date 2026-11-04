//! First-root admission through the originating host on the owned private pipe.
//! No Server credential or lease ID crosses this boundary.
use runmat_execution::{resource::ResourceUsage, ProgramExecutionAssignment};
use runmat_runtime::analysis::fea_run_admission::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub(crate) const MAX_FRAME_BYTES: u64 = 16 * 1024;
pub(crate) const STORAGE_BYTES: u64 =
    MAX_FRAME_BYTES * 4 + 4096 + FEA_DEADLINE_MONITOR_STACK_BYTES as u64;
pub type Provider = Result<Arc<dyn FeaRunAdmission>, FeaRunDenied>;
pub fn capture() -> Provider {
    capture_fea_run_admission()
}

#[derive(Clone, Copy)]
pub(crate) enum StorageRole {
    Proxy,
    FirstRoot,
    Delegated,
}
/// One quote for automatic task admission and the actual private launch/root
/// checks. Inherited roots have no second usage-reporter thread.
pub(crate) fn storage_bytes(invocation_bytes: u64, role: StorageRole) -> Result<u64, String> {
    let component = match role {
        StorageRole::Proxy => STORAGE_BYTES,
        StorageRole::FirstRoot => STORAGE_BYTES
            .checked_add(FEA_HEARTBEAT_STACK_BYTES as u64)
            .ok_or("FEA control storage overflow")?,
        StorageRole::Delegated => crate::fea_delegation::STORAGE_BYTES,
    };
    invocation_bytes
        .checked_add(component)
        .and_then(|bytes| {
            bytes.checked_add(runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES)
        })
        .ok_or_else(|| "FEA control storage overflow".into())
}
pub(crate) fn automatic_storage_bytes() -> Result<u64, String> {
    let role = if FeaRunScope::current_delegation_source()
        .map_err(|error| error.to_string())?
        .is_some()
    {
        StorageRole::Delegated
    } else if capture().is_ok() {
        StorageRole::FirstRoot
    } else {
        StorageRole::Proxy
    };
    storage_bytes(0, role)
}
pub(crate) fn is_call(bytes: &[u8]) -> bool {
    if bytes.len() as u64 > MAX_FRAME_BYTES {
        return false;
    }
    #[derive(Deserialize)]
    #[serde(tag = "message", rename_all = "snake_case")]
    enum Probe {
        FeaAdmissionCall {},
    }
    matches!(
        serde_json::from_slice::<Probe>(bytes),
        Ok(Probe::FeaAdmissionCall {})
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Start {
        request: FeaRunRequest,
    },
    Heartbeat {
        root: u64,
        usage: FeaRunUsage,
    },
    RequestGrant {
        root: u64,
        usage: FeaRunUsage,
        required: runmat_simulation_usage::OperationCount,
    },
    Settle {
        root: u64,
        outcome: FeaRunOutcome,
        usage: FeaRunUsage,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub assignment: ProgramExecutionAssignment,
    pub sequence: u64,
    pub operation: Operation,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reply {
    Budget { budget: FeaRunBudget },
    Settled,
    Denied { denied: FeaRunDenied },
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(u64::MAX, |time| {
            time.as_millis().min(u128::from(u64::MAX)) as u64
        })
}
fn usage(cpu: u64, wall: u64) -> ResourceUsage {
    ResourceUsage {
        wall_millis: wall,
        cpu_core_millis: (u128::from(cpu) * u128::from(wall))
            .div_ceil(1000)
            .min(u128::from(u64::MAX)) as u64,
        ..Default::default()
    }
}
mod owner;
mod worker;
pub use owner::{Close, Parent};
pub use worker::{Client, Replies, WorkerAdmission};

/// Decode only the existing bounded host-call frame, without widening test protocol.
pub fn decode_call(bytes: &[u8]) -> Result<Option<Call>, String> {
    if !is_call(bytes) {
        return Ok(None);
    }
    match serde_json::from_slice::<crate::protocol::WorkerProcessMessage>(bytes)
        .map_err(|e| e.to_string())?
    {
        crate::protocol::WorkerProcessMessage::FeaAdmissionCall { call } => Ok(Some(call)),
        _ => Err("invalid FEA admission call envelope".into()),
    }
}
pub fn encode_reply(sequence: u64, reply: Reply) -> Result<Vec<u8>, String> {
    runmat_canonical_codec::encode_json_bounded(
        &crate::protocol::WorkerDriverMessage::FeaAdmissionReply { sequence, reply },
        MAX_FRAME_BYTES,
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    #[test]
    fn typed_probe_only_scans_bounded_admission_calls() {
        assert!(is_call(br#"{"message":"fea_admission_call","call":{}}"#));
        assert!(!is_call(br#"{"message":"completed"}"#));
        assert!(!is_call(&vec![b' '; MAX_FRAME_BYTES as usize + 1]));
    }
}
