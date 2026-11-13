use runmat_execution::identity::AttemptId;
use runmat_execution::identity::{PoolId, WorkerId};
use runmat_execution::resource::Capability;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_runner::{AttemptReport, WorkerSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use runmat_execution_transport_native::control::DriverArtifactDownload;
use runmat_execution_transport_native::transfer::ObjectChunk;

use super::CheckpointedAttempt;
use super::{RemoteAttempt, RemoteBundleReceipt};

#[cfg(test)]
use super::protocol_schema::REMOTE_WORKER_PROTOCOL_V4;
pub use super::protocol_schema::REMOTE_WORKER_PROTOCOL_VERSION;

/// Fixed-size counters retained with the original worker reservation. The
/// allocation and leased devices remain in the exact request, not a second copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetiredAttemptUsage {
    pub wall_millis: u64,
    pub artifact_bytes: u64,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

impl RetiredAttemptUsage {
    pub(super) fn from_accounting(
        resources: &runmat_execution_runner::task::AttemptResourceAccounting,
    ) -> Self {
        Self {
            wall_millis: resources.resources.usage.wall_millis,
            artifact_bytes: resources.resources.artifact_bytes,
            egress_bytes: resources.resources.usage.egress_bytes,
            relay_bytes: resources.resources.usage.relay_bytes,
        }
    }

    pub(super) fn accounting(
        self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> Result<runmat_execution_runner::task::AttemptResourceAccounting, String> {
        runmat_execution_runner::task::AttemptResourceAccounting::for_attempt(
            request,
            self.wall_millis,
            self.artifact_bytes,
            self.egress_bytes,
            self.relay_bytes,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteWorkerRequest {
    pub schema_version: u16,
    pub correlation_id: String,
    pub driver_fence: u64,
    pub command: RemoteWorkerCommand,
}

impl RemoteWorkerRequest {
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.schema_version != REMOTE_WORKER_PROTOCOL_VERSION {
            return Err(format!(
                "unsupported remote worker protocol {}; expected {}; rebuild or upgrade the remote worker",
                self.schema_version, REMOTE_WORKER_PROTOCOL_VERSION
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RemoteWorkerCommand {
    Describe,
    InstallBundle {
        bundle_digest: Digest,
        bundle: Vec<u8>,
    },
    ActivateBundle {
        bundle_digest: Digest,
    },
    BeginGraphProgram {
        attempt_id: AttemptId,
        encrypted_manifest: Vec<u8>,
        maximum_program_bytes: u64,
    },
    StageGraphProgramChunk {
        attempt_id: AttemptId,
        program_digest: Digest,
        ordinal: u32,
        download: DriverArtifactDownload,
    },
    FinishGraphProgram {
        attempt_id: AttemptId,
        program_digest: Digest,
    },
    PutValue {
        reference: ValueRef,
        encoded: Vec<u8>,
    },
    ProbeObject {
        reference: ValueRef,
    },
    PutObjectChunk {
        reference: ValueRef,
        chunk: ObjectChunk,
    },
    GetObjectChunk {
        reference: ValueRef,
        offset: u64,
        maximum_bytes: u32,
    },
    Execute {
        attempt: Box<RemoteAttempt>,
        #[serde(deserialize_with = "runmat_execution::console::deserialize_budget")]
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
    },
    CompleteCollective {
        attempt_id: AttemptId,
        context: runmat_execution::SpmdTaskContext,
        id: runmat_types::CollectiveId,
        sequence: runmat_execution::CollectiveSequence,
        result: crate::protocol::CollectiveProcessResult,
    },
    CompleteFeaAdmission {
        attempt_id: AttemptId,
        sequence: u64,
        reply: crate::fea_proxy::Reply,
    },
    Cancel {
        attempt_id: AttemptId,
        reason: runmat_execution::CancellationReason,
    },
    QueryPhysicalRetirement {
        attempt: Box<runmat_execution_runner::AttemptRequest>,
    },
    AcknowledgePhysicalRetirement {
        attempt: Box<runmat_execution_runner::AttemptRequest>,
    },
    PruneDurableCheckpoint {
        sequence: u64,
        attempts: Vec<CheckpointedAttempt>,
    },
    Drain,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteWorkerReply {
    pub schema_version: u16,
    pub correlation_id: String,
    pub outcome: RemoteWorkerOutcome,
}

/// Software capabilities issued by the worker executable, not by an
/// allocation request or node-agent inventory.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerReady {
    pub run_identity: String,
    pub session_id: [u8; 16],
    pub driver_fence: u64,
    pub worker_id: WorkerId,
    pub pool_id: PoolId,
    pub capabilities: BTreeSet<Capability>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RemoteWorkerOutcome {
    WorkerDescription {
        worker: WorkerSpec,
    },
    WorkerReady {
        ready: WorkerReady,
    },
    BundleStored {
        receipt: RemoteBundleReceipt,
    },
    GraphProgramStaged {
        receipt: super::RemoteGraphProgramReceipt,
    },
    GraphProgramPrepared {
        program_digest: Digest,
        program_length: u64,
        chunk_count: u32,
    },
    ValueStored {
        receipt: super::RemoteValueReceipt,
    },
    ObjectPosition {
        receipt: super::RemoteObjectReceipt,
    },
    ObjectChunk {
        chunk: ObjectChunk,
        complete: bool,
    },
    Progress {
        attempt_id: AttemptId,
        progress: crate::ProgramProgress,
    },
    CollectiveRequest {
        attempt_id: AttemptId,
        request: runmat_execution::CollectiveRequest,
    },
    FeaAdmissionCall {
        attempt_id: AttemptId,
        call: crate::fea_proxy::Call,
    },
    Attempt {
        #[serde(deserialize_with = "runmat_execution::console::deserialize_source_identity")]
        delegated_console_root: Option<runmat_execution::console::ConsoleSourceIdentity>,
        report: AttemptReport,
        #[serde(deserialize_with = "runmat_execution::console::deserialize_transcript")]
        console: Option<Box<runmat_execution::console::ConsoleTranscript>>,
        #[serde(deserialize_with = "runmat_execution::console::deserialize_scopes")]
        console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
        #[serde(deserialize_with = "runmat_execution::console::deserialize_budget")]
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
    },
    AdmissionDeferred {
        attempt_id: AttemptId,
        worker_id: WorkerId,
        driver_fence: u64,
    },
    PhysicalRetirement {
        run_identity: String,
        session_id: [u8; 16],
        driver_fence: u64,
        worker_id: WorkerId,
        pool_id: PoolId,
        attempt_id: AttemptId,
        request_digest: Digest,
        usage: Option<RetiredAttemptUsage>,
    },
    RetirementUnavailable,
    Acknowledged,
    Rejected {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::{AttemptId, GangId, PoolId};
    use runmat_execution::value::{InlineValue, ValuePayload};
    use runmat_execution::{
        CollectiveInvocation, CollectiveRequest, CollectiveSequence, ExecutionScopeId, GangHandle,
        PoolHandle, SpmdTaskContext,
    };
    use runmat_execution_artifact::encryption::RunKeyMaterial;
    use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
    use runmat_types::{
        CollectiveId, LabCount, LabRank, ParallelRegionId, ProgramFunctionId, RegionId,
    };

    use super::*;

    #[test]
    fn frozen_remote_worker_v4_request_is_rejected_before_dispatch() {
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_V4,
            correlation_id: "stale-request".into(),
            driver_fence: 1,
            command: RemoteWorkerCommand::Drain,
        };
        let error = request.validate_schema().unwrap_err();
        assert!(error.contains(&format!(
            "unsupported remote worker protocol 4; expected {REMOTE_WORKER_PROTOCOL_VERSION}"
        )));
        assert!(error.contains("rebuild or upgrade"));
    }

    fn context() -> SpmdTaskContext {
        let scope_id = ExecutionScopeId::derive(&[b"scope"]);
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(7),
            ordinal: 3,
        });
        SpmdTaskContext {
            gang: GangHandle {
                id: GangId::derive(&[b"gang"]),
                scope_id,
                generation: 11,
                pool: PoolHandle {
                    id: PoolId::derive(&[b"pool"]),
                    scope_id,
                    generation: 5,
                },
                labs: LabCount(2),
            },
            region,
            rank: LabRank(2),
        }
    }

    #[test]
    fn collective_completion_round_trips_every_typed_identity() {
        let context = context();
        let id = CollectiveId {
            region: context.region,
            ordinal: 13,
        };
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: "collective-completion".into(),
            driver_fence: 19,
            command: RemoteWorkerCommand::CompleteCollective {
                attempt_id: AttemptId::derive(&[b"attempt"]),
                context: context.clone(),
                id,
                sequence: CollectiveSequence(17),
                result: crate::protocol::CollectiveProcessResult::Completed {
                    response: runmat_execution::CollectiveResponse::Complete,
                },
            },
        };

        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: RemoteWorkerRequest = serde_json::from_slice(&encoded).unwrap();
        let RemoteWorkerCommand::CompleteCollective {
            attempt_id,
            context: decoded_context,
            id: decoded_id,
            sequence,
            result,
        } = decoded.command
        else {
            panic!("collective completion changed command kind");
        };
        assert_eq!(attempt_id, AttemptId::derive(&[b"attempt"]));
        assert_eq!(decoded_context, context);
        assert_eq!(decoded_id, id);
        assert_eq!(sequence, CollectiveSequence(17));
        assert!(matches!(
            result,
            crate::protocol::CollectiveProcessResult::Completed {
                response: runmat_execution::CollectiveResponse::Complete
            }
        ));
    }

    #[test]
    fn collective_completion_rejects_unknown_wire_fields() {
        let context = context();
        let request = RemoteWorkerRequest {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: "collective-completion".into(),
            driver_fence: 19,
            command: RemoteWorkerCommand::CompleteCollective {
                attempt_id: AttemptId::derive(&[b"attempt"]),
                context: context.clone(),
                id: CollectiveId {
                    region: context.region,
                    ordinal: 13,
                },
                sequence: CollectiveSequence(17),
                result: crate::protocol::CollectiveProcessResult::Completed {
                    response: runmat_execution::CollectiveResponse::Complete,
                },
            },
        };
        let mut encoded = serde_json::to_value(request).unwrap();
        encoded
            .get_mut("command")
            .and_then(serde_json::Value::as_object_mut)
            .unwrap()
            .insert("untypedAlias".into(), serde_json::json!("rank-2"));

        assert!(serde_json::from_value::<RemoteWorkerRequest>(encoded).is_err());
    }

    #[test]
    fn collective_values_are_opaque_to_the_frame_route() {
        const PRIVATE_VALUE: &str = "private-rank-value-6f7acb";
        let context = context();
        let reply = RemoteWorkerReply {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: "collective-request".into(),
            outcome: RemoteWorkerOutcome::CollectiveRequest {
                attempt_id: AttemptId::derive(&[b"attempt"]),
                request: CollectiveRequest {
                    context: context.clone(),
                    id: CollectiveId {
                        region: context.region,
                        ordinal: 9,
                    },
                    sequence: CollectiveSequence(4),
                    invocation: CollectiveInvocation::Broadcast {
                        root: LabRank(1),
                        value: Some(ValuePayload::Inline(Box::new(InlineValue::String(
                            PRIVATE_VALUE.into(),
                        )))),
                    },
                },
            },
        };
        let plaintext = serde_json::to_vec(&reply).unwrap();
        assert!(plaintext
            .windows(PRIVATE_VALUE.len())
            .any(|window| window == PRIVATE_VALUE.as_bytes()));

        let key = RunKeyMaterial::from_entropy([23; 32]).unwrap();
        let limits = FrameLimits::default();
        let mut sender =
            EncryptedFrameSession::new("private-run", [29; 16], "worker-to-driver", 1, key.clone())
                .unwrap();
        let mut receiver =
            EncryptedFrameSession::new("private-run", [29; 16], "worker-to-driver", 1, key)
                .unwrap();
        let frame = sender
            .seal_with_entropy(FrameKind::Control, &plaintext, [31; 32], limits)
            .unwrap();

        assert!(!frame
            .payload
            .windows(PRIVATE_VALUE.len())
            .any(|window| window == PRIVATE_VALUE.as_bytes()));
        let decoded: RemoteWorkerReply =
            serde_json::from_slice(&receiver.open(&frame, limits).unwrap()).unwrap();
        let RemoteWorkerOutcome::CollectiveRequest { request, .. } = decoded.outcome else {
            panic!("encrypted collective changed outcome kind");
        };
        let CollectiveInvocation::Broadcast { value, .. } = request.invocation else {
            panic!("encrypted collective changed invocation kind");
        };
        assert!(matches!(
            value,
            Some(ValuePayload::Inline(value))
                if matches!(value.as_ref(), InlineValue::String(text) if text.as_str() == Some(PRIVATE_VALUE))
        ));
    }
}
