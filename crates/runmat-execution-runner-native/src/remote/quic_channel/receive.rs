//! Authenticate and deliver replies; retain the original reason a channel closes.
use super::*;

pub(super) async fn run(
    route: &dyn RemoteFrameRoute,
    receiver: &mut EncryptedFrameSession,
    limits: FrameLimits,
    pending: &Mutex<ReplyRegistry>,
    progress: &Mutex<
        HashMap<runmat_execution::identity::AttemptId, VecDeque<crate::ProgramProgress>>,
    >,
) -> NativeExecutionResult<()> {
    let mut progress_sequences = HashMap::new();
    loop {
        let frame = route.receive().await?;
        let plaintext = receiver.open(&frame, limits).map_err(protocol)?;
        let reply: RemoteWorkerReply = serde_json::from_slice(&plaintext).map_err(protocol)?;
        if reply.schema_version != REMOTE_WORKER_PROTOCOL_VERSION {
            return Err(protocol("remote worker reply has an unsupported schema"));
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
            }
            let _ = request.sender.send(Ok(reply));
        }
    }
}
