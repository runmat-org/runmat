use runmat_execution::identity::AttemptId;
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_runner::AttemptReport;
use serde::{Deserialize, Serialize};

use runmat_execution_transport_native::transfer::ObjectChunk;

use super::{RemoteAttempt, RemoteBundleReceipt};

pub const REMOTE_WORKER_PROTOCOL_V4: u16 = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteWorkerRequest {
    pub schema_version: u16,
    pub correlation_id: String,
    pub driver_fence: u64,
    pub command: RemoteWorkerCommand,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RemoteWorkerCommand {
    InstallBundle {
        bundle_digest: Digest,
        bundle: Vec<u8>,
    },
    ActivateBundle {
        bundle_digest: Digest,
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
    },
    CompleteCollective {
        attempt_id: AttemptId,
        context: runmat_execution::SpmdTaskContext,
        id: runmat_types::CollectiveId,
        sequence: runmat_execution::CollectiveSequence,
        result: crate::protocol::CollectiveProcessResult,
    },
    Cancel {
        attempt_id: AttemptId,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RemoteWorkerOutcome {
    BundleStored {
        receipt: RemoteBundleReceipt,
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
    Attempt {
        report: AttemptReport,
    },
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
            schema_version: REMOTE_WORKER_PROTOCOL_V4,
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
            schema_version: REMOTE_WORKER_PROTOCOL_V4,
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
            schema_version: REMOTE_WORKER_PROTOCOL_V4,
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
                if matches!(value.as_ref(), InlineValue::String(text) if text == PRIVATE_VALUE)
        ));
    }
}
