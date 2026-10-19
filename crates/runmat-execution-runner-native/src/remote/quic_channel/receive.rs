//! Authenticate and deliver replies; retain the original reason a channel closes.
use super::*;
use runmat_execution::identity::{AttemptId, PoolId, WorkerId};
use runmat_execution::CollectiveRequest;
use std::collections::BTreeSet;

/// The exact route identity an authenticated ReadyV1 reply must carry before
/// its software capabilities are admitted.
pub(super) struct ReadyExpectation {
    pub run_identity: String,
    pub session_id: [u8; 16],
    pub driver_fence: u64,
    pub worker_id: WorkerId,
    pub pool_id: PoolId,
    pub capabilities: Arc<Mutex<Option<BTreeSet<Capability>>>>,
}

pub(super) struct Registries<'a> {
    pub pending: &'a Mutex<ReplyRegistry>,
    pub progress: &'a Mutex<HashMap<AttemptId, VecDeque<crate::ProgramProgress>>>,
    pub collectives: &'a Mutex<HashMap<AttemptId, VecDeque<CollectiveRequest>>>,
}

pub(super) async fn run(
    route: &dyn RemoteFrameRoute,
    receiver: &mut EncryptedFrameSession,
    limits: FrameLimits,
    ready: &ReadyExpectation,
    registries: Registries<'_>,
) -> NativeExecutionResult<()> {
    let Registries {
        pending,
        progress,
        collectives,
    } = registries;
    let mut progress_sequences = HashMap::new();
    loop {
        let frame = route.receive().await?;
        let plaintext = receiver.open(&frame, limits).map_err(protocol)?;
        if super::super::protocol_schema::admit_remote_worker_bytes(&plaintext).is_err() {
            continue;
        }
        let reply: RemoteWorkerReply = serde_json::from_slice(&plaintext).map_err(protocol)?;
        if reply.schema_version != REMOTE_WORKER_PROTOCOL_VERSION {
            return Err(protocol("remote worker reply has an unsupported schema"));
        }
        if let RemoteWorkerOutcome::WorkerReady { ready: issued } = &reply.outcome {
            if frame.kind != FrameKind::Control
                || reply.correlation_id != "worker-ready-v1"
                || issued.run_identity != ready.run_identity
                || issued.session_id != ready.session_id
                || issued.driver_fence != ready.driver_fence
                || issued.worker_id != ready.worker_id
                || issued.pool_id != ready.pool_id
            {
                return Err(protocol("remote worker readiness differs from its route"));
            }
            let mut state = ready.capabilities.lock().expect("worker readiness poisoned");
            if state.is_some() {
                return Err(protocol("remote worker issued readiness twice"));
            }
            *state = Some(issued.capabilities.clone());
            continue;
        }
        if let RemoteWorkerOutcome::Progress {
            attempt_id,
            progress: event,
        } = &reply.outcome
        {
            event.validate().map_err(protocol)?;
            // Late progress cannot recreate closed request state or grow queues
            // for arbitrary attempts. Live activity belongs to its exact task.
            let pending = pending.lock().expect("remote reply registry poisoned");
            let Some(request) = pending.get(&reply.correlation_id) else {
                continue;
            };
            if request.attempt_id != Some(*attempt_id) || frame.kind != request.frame_kind {
                return Err(protocol(
                    "remote worker progress differs from its original request",
                ));
            }
            if progress_sequences
                .insert(*attempt_id, event.sequence)
                .is_some_and(|previous| event.sequence <= previous)
            {
                return Err(protocol("remote worker progress is not strictly monotone"));
            }
            let mut queues = progress.lock().expect("remote progress registry poisoned");
            let queue = queues.entry(*attempt_id).or_default();
            if queue.len() == MAX_BUFFERED_PROGRESS {
                queue.pop_front();
            }
            queue.push_back(event.clone());
            request.activity.send_replace(tokio::time::Instant::now());
            continue;
        }
        if let RemoteWorkerOutcome::CollectiveRequest {
            attempt_id,
            request: collective,
        } = &reply.outcome
        {
            collective.validate().map_err(protocol)?;
            // A collective belongs to the exact pending attempt that issued it;
            // waiting on peers is live activity for that attempt.
            let pending = pending.lock().expect("remote reply registry poisoned");
            let Some(request) = pending.get(&reply.correlation_id) else {
                continue;
            };
            if request.attempt_id != Some(*attempt_id) || frame.kind != request.frame_kind {
                return Err(protocol(
                    "remote worker collective differs from its original request",
                ));
            }
            collectives
                .lock()
                .expect("remote collective registry poisoned")
                .entry(*attempt_id)
                .or_default()
                .push_back(collective.clone());
            request.activity.send_replace(tokio::time::Instant::now());
            continue;
        }
        let mut pending = pending.lock().expect("remote reply registry poisoned");
        if pending
            .get(&reply.correlation_id)
            .is_some_and(|request| request.frame_kind != frame.kind)
        {
            return Err(protocol("remote worker reply has the wrong frame kind"));
        }
        if let Some(request) = pending.remove(&reply.correlation_id) {
            if let Some(attempt_id) = request.attempt_id {
                progress_sequences.remove(&attempt_id);
                collectives
                    .lock()
                    .expect("remote collective registry poisoned")
                    .remove(&attempt_id);
            }
            let _ = request.sender.send(Ok(reply));
        }
    }
}
