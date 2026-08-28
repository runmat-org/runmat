use runmat_execution::identity::AttemptId;
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use tokio::sync::Mutex;

use super::protocol::{
    RemoteWorkerCommand, RemoteWorkerOutcome, RemoteWorkerReply, REMOTE_WORKER_PROTOCOL_V3,
};
use super::route::RemoteFrameRoute;
use crate::{NativeExecutionError, NativeExecutionResult, ProgramProgress};

pub(super) async fn reply(
    connection: &dyn RemoteFrameRoute,
    sender: &Mutex<EncryptedFrameSession>,
    limits: FrameLimits,
    reply: RemoteWorkerReply,
) -> NativeExecutionResult<()> {
    reply_kind(connection, sender, limits, FrameKind::Control, reply).await
}

pub(super) async fn reply_kind(
    connection: &dyn RemoteFrameRoute,
    sender: &Mutex<EncryptedFrameSession>,
    limits: FrameLimits,
    kind: FrameKind,
    reply: RemoteWorkerReply,
) -> NativeExecutionResult<()> {
    let plaintext = serde_json::to_vec(&reply).map_err(protocol)?;
    let frame = sender
        .lock()
        .await
        .seal(kind, &plaintext, limits)
        .map_err(protocol)?;
    connection.send(frame).await
}

pub(super) async fn reply_progress(
    connection: &dyn RemoteFrameRoute,
    sender: &Mutex<EncryptedFrameSession>,
    limits: FrameLimits,
    correlation_id: &str,
    attempt_id: AttemptId,
    progress: ProgramProgress,
) -> NativeExecutionResult<()> {
    reply(
        connection,
        sender,
        limits,
        RemoteWorkerReply {
            schema_version: REMOTE_WORKER_PROTOCOL_V3,
            correlation_id: correlation_id.into(),
            outcome: RemoteWorkerOutcome::Progress {
                attempt_id,
                progress,
            },
        },
    )
    .await
}

pub(super) fn command_frame_kind(command: &RemoteWorkerCommand) -> FrameKind {
    match command {
        RemoteWorkerCommand::ProbeObject { .. }
        | RemoteWorkerCommand::PutObjectChunk { .. }
        | RemoteWorkerCommand::GetObjectChunk { .. } => FrameKind::Artifact,
        _ => FrameKind::Control,
    }
}

pub(super) fn acknowledged(correlation_id: String) -> RemoteWorkerReply {
    RemoteWorkerReply {
        schema_version: REMOTE_WORKER_PROTOCOL_V3,
        correlation_id,
        outcome: RemoteWorkerOutcome::Acknowledged,
    }
}

pub(super) fn rejected(correlation_id: String, message: impl Into<String>) -> RemoteWorkerReply {
    RemoteWorkerReply {
        schema_version: REMOTE_WORKER_PROTOCOL_V3,
        correlation_id,
        outcome: RemoteWorkerOutcome::Rejected {
            message: message.into(),
        },
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
