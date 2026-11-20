//! Native transport for solver progress, shared by local and remote workers.
//!
//! Inside a local analysis worker process, [`attach_worker_writer`] connects solver progress to
//! the parent: each event is rate limited, sent as a [`ProgramProgress`] frame
//! ([`FEA_PROGRESS_MEDIA_TYPE`]) and decoded by the parent with [`decode`].
use std::sync::{Arc, Mutex, OnceLock};

use runmat_analysis_fea::{
    FeaCancellationToken, FeaProgressContext, FeaProgressContextGuard, FeaProgressEvent,
};
use runmat_execution::identity::AttemptId;
use runmat_process_host::ipc::private_control::ControlFrameWriter;

use crate::protocol::{ProgramProgress, WorkerProcessMessage, NATIVE_WORKER_MESSAGE_SCHEMA_V2};

pub(crate) const FEA_PROGRESS_MEDIA_TYPE: &str = "application/vnd.runmat.fea-progress+json";
pub(crate) const FEA_PROGRESS_VALUE_SCHEMA: &str = "runmat.fea.progress_event.v1";
/// Largest encoded event a worker sends; longer events are dropped.
const MAXIMUM_EVENT_BYTES: usize = 16 * 1024;

struct WorkerProgressWriter {
    writer: ControlFrameWriter,
    max_message_bytes: u64,
    state: Mutex<WriterState>,
}

#[derive(Default)]
struct WriterState {
    sequence: u64,
    limiter: runmat_analysis_fea::progress::ProgressRateLimiter,
}

static WORKER_WRITER: OnceLock<WorkerProgressWriter> = OnceLock::new();

/// Sends this worker process's solver progress to its parent. A worker serves one launch, so the
/// first attachment holds for the life of the process.
pub(crate) fn attach_worker_writer(writer: ControlFrameWriter, max_message_bytes: u32) {
    let _ = WORKER_WRITER.set(WorkerProgressWriter {
        writer,
        max_message_bytes: u64::from(max_message_bytes),
        state: Mutex::new(WriterState::default()),
    });
}

fn forward_to_parent(event: &FeaProgressEvent) {
    let Some(target) = WORKER_WRITER.get() else {
        return;
    };
    let mut state = target
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !state.limiter.admit(event, std::time::Instant::now()) {
        return;
    }
    let Ok(payload) = serde_json::to_vec(event) else {
        return;
    };
    if payload.len() > MAXIMUM_EVENT_BYTES {
        return;
    }
    state.sequence += 1;
    let message = WorkerProcessMessage::Progress {
        progress: ProgramProgress {
            schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V2,
            sequence: state.sequence,
            media_type: FEA_PROGRESS_MEDIA_TYPE.into(),
            value_schema: FEA_PROGRESS_VALUE_SCHEMA.into(),
            payload,
        },
    };
    // The sequence is assigned and written under one lock, so frames arrive in order. A lost
    // parent surfaces through the terminal frame; progress never fails the solve.
    if let Ok(frame) =
        runmat_canonical_codec::encode_json_bounded(&message, target.max_message_bytes)
    {
        let _ = target.writer.write_blocking(frame);
    }
}

/// The FEA progress event a worker frame carries, or `None` for other progress media.
pub(crate) fn decode(progress: &ProgramProgress) -> Option<FeaProgressEvent> {
    (progress.media_type == FEA_PROGRESS_MEDIA_TYPE
        && progress.value_schema == FEA_PROGRESS_VALUE_SCHEMA)
        .then(|| serde_json::from_slice(&progress.payload).ok())
        .flatten()
}

pub(crate) fn install(
    cancellation: FeaCancellationToken,
    attempt_id: Option<AttemptId>,
) -> FeaProgressContextGuard {
    let recorded_attempt = attempt_id.map(|id| id.to_string());
    runmat_analysis_fea::replace_fea_progress_context(FeaProgressContext {
        handler: Some(Arc::new(move |event| {
            tracing::info!(
                event_code = "analysis.solver.progress",
                attempt_id = recorded_attempt.as_deref().unwrap_or(""),
                operation = %event.operation,
                phase = ?event.phase,
                status = ?event.status,
                current = ?event.current,
                "analysis solver progress: {}",
                event.message
            );
            forward_to_parent(&event);
        })),
        cancellation,
        attempt_id,
    })
}

#[cfg(test)]
mod tests;
