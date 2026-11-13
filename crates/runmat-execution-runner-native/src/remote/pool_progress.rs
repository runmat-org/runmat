use std::collections::VecDeque;
use std::sync::Mutex;

use runmat_execution::identity::AttemptId;
use runmat_execution::SpmdTaskContext;
use runmat_execution_runner::AttemptSuccess;
use tokio::sync::oneshot;

use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult, ProgramProgress};

const MAX_BUFFERED_PROGRESS: usize = 256;

#[derive(Default)]
struct State {
    active_attempt: Option<AttemptId>,
    last_attempt_sequence: u64,
    next_task_sequence: u64,
    queue: VecDeque<ProgramProgress>,
}

#[derive(Default)]
pub(super) struct RemoteProgressBuffer(Mutex<State>);

pub(super) struct RemoteCollectiveAssignment {
    broker: std::sync::Arc<crate::driver::collective::ProcessCollectiveBroker>,
    context: SpmdTaskContext,
}

impl RemoteCollectiveAssignment {
    pub(super) fn new(
        broker: std::sync::Arc<crate::driver::collective::ProcessCollectiveBroker>,
        context: SpmdTaskContext,
    ) -> Self {
        Self { broker, context }
    }
}

pub struct RemoteTaskCompletion {
    receiver: oneshot::Receiver<(u64, Result<AttemptSuccess, crate::NativeProgramFailure>)>,
    progress: std::sync::Arc<RemoteProgressBuffer>,
    origin: RemoteCompletionOrigin,
    original_graph_wall_millis: Option<u64>,
}

#[derive(Clone, Copy)]
enum RemoteCompletionOrigin {
    Executing,
    ExistingInvocation(runmat_execution::TaskId),
}

impl RemoteTaskCompletion {
    pub(super) fn new(
        receiver: oneshot::Receiver<(u64, Result<AttemptSuccess, crate::NativeProgramFailure>)>,
        progress: std::sync::Arc<RemoteProgressBuffer>,
    ) -> Self {
        Self {
            receiver,
            progress,
            origin: RemoteCompletionOrigin::Executing,
            original_graph_wall_millis: None,
        }
    }

    pub(super) fn resolved_committed(
        task: runmat_execution::TaskId,
        result: AttemptSuccess,
        original_graph_wall_millis: u64,
    ) -> Self {
        let (sender, receiver) = oneshot::channel();
        let _ = sender.send((0, Ok(result)));
        Self {
            receiver,
            progress: std::sync::Arc::new(RemoteProgressBuffer::default()),
            origin: RemoteCompletionOrigin::ExistingInvocation(task),
            original_graph_wall_millis: Some(original_graph_wall_millis),
        }
    }

    pub(super) fn resolved_pending(task: runmat_execution::TaskId, result: AttemptSuccess) -> Self {
        let (sender, receiver) = oneshot::channel();
        let _ = sender.send((0, Ok(result)));
        Self {
            receiver,
            progress: std::sync::Arc::new(RemoteProgressBuffer::default()),
            origin: RemoteCompletionOrigin::ExistingInvocation(task),
            original_graph_wall_millis: None,
        }
    }

    pub(super) fn resume_existing(mut self, task: runmat_execution::TaskId) -> Self {
        self.origin = RemoteCompletionOrigin::ExistingInvocation(task);
        self
    }

    /// Only exact immutable-task replay can issue this origin. It may identify
    /// a restored pending task; callers must await accepted success before
    /// using it as provenance for a reused result.
    pub(crate) fn reused_invocation(&self) -> Option<runmat_execution::TaskId> {
        match self.origin {
            RemoteCompletionOrigin::Executing => None,
            RemoteCompletionOrigin::ExistingInvocation(task) => Some(task),
        }
    }

    pub(crate) fn original_graph_wall_millis(&self) -> Option<u64> {
        self.original_graph_wall_millis
    }

    pub async fn wait(self) -> Result<AttemptSuccess, crate::NativeProgramFailure> {
        self.wait_ordered().await.1
    }

    pub(super) async fn wait_ordered(
        self,
    ) -> (u64, Result<AttemptSuccess, crate::NativeProgramFailure>) {
        self.receiver.await.unwrap_or_else(|_| {
            (
                u64::MAX,
                Err(crate::NativeProgramFailure::Infrastructure(
                    "remote task completion channel closed".into(),
                )),
            )
        })
    }

    pub async fn wait_with_progress(
        self,
    ) -> (
        Result<AttemptSuccess, crate::NativeProgramFailure>,
        Vec<ProgramProgress>,
    ) {
        let progress = std::sync::Arc::clone(&self.progress);
        let (_, result) = self.wait_ordered().await;
        (result, progress.drain())
    }

    pub fn drain_progress(&self) -> Vec<ProgramProgress> {
        self.progress.drain()
    }
}

impl RemoteProgressBuffer {
    pub(super) fn drain(&self) -> Vec<ProgramProgress> {
        self.0
            .lock()
            .expect("remote task progress poisoned")
            .queue
            .drain(..)
            .collect()
    }

    fn append(
        &self,
        channel: &dyn RemoteWorkerChannel,
        attempt_id: AttemptId,
    ) -> NativeExecutionResult<()> {
        let mut state = self.0.lock().expect("remote task progress poisoned");
        if state.active_attempt != Some(attempt_id) {
            state.active_attempt = Some(attempt_id);
            state.last_attempt_sequence = 0;
        }
        for mut event in channel.drain_progress(attempt_id) {
            event.validate().map_err(NativeExecutionError::Protocol)?;
            if event.sequence <= state.last_attempt_sequence {
                return Err(NativeExecutionError::Protocol(
                    "remote attempt progress sequence is not monotone".into(),
                ));
            }
            state.last_attempt_sequence = event.sequence;
            state.next_task_sequence =
                state.next_task_sequence.checked_add(1).ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "remote task progress sequence overflowed".into(),
                    )
                })?;
            event.sequence = state.next_task_sequence;
            if state.queue.len() == MAX_BUFFERED_PROGRESS {
                state.queue.pop_front();
            }
            state.queue.push_back(event);
        }
        Ok(())
    }
}

pub(super) async fn execute(
    channel: &dyn RemoteWorkerChannel,
    attempt: RemoteAttempt,
    progress: Option<&RemoteProgressBuffer>,
    collectives: Option<RemoteCollectiveAssignment>,
    console_budget: Option<runmat_execution::console::ConsoleBudget>,
    fea_provider: &crate::fea_proxy::Provider,
) -> NativeExecutionResult<runmat_execution_runner::port::BackendReport> {
    let attempt_id = attempt.scheduling.id;
    let scheduling = attempt.scheduling.clone();
    let mut fea_parent = None;
    let execution = channel.execute_observed(attempt, console_budget);
    tokio::pin!(execution);
    loop {
        tokio::select! {
            result = &mut execution => {
                if let Some(progress) = progress {
                    progress.append(channel, attempt_id)?;
                }
                forward_collectives(channel, attempt_id, collectives.as_ref()).await?;
                forward_fea(channel, &scheduling, fea_provider, &mut fea_parent).await?;
                if matches!(&result, Ok(report) if matches!(&report.report, runmat_execution_runner::AttemptReport::Succeeded { .. })) {
                    if let Some(parent) = &fea_parent {
                        parent.terminal_check().map_err(|error| {
                            NativeExecutionError::Protocol(format!("remote FEA parent did not settle before successful attempt: {error}"))
                        })?;
                    }
                }
                return result;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                if let Some(progress) = progress {
                    progress.append(channel, attempt_id)?;
                }
                forward_collectives(channel, attempt_id, collectives.as_ref()).await?;
                forward_fea(channel, &scheduling, fea_provider, &mut fea_parent).await?;
            }
        }
    }
}

async fn forward_fea(
    channel: &dyn RemoteWorkerChannel,
    scheduling: &runmat_execution_runner::AttemptRequest,
    provider: &crate::fea_proxy::Provider,
    parent: &mut Option<std::sync::Arc<crate::fea_proxy::Parent>>,
) -> NativeExecutionResult<()> {
    for call in channel.drain_fea_calls(scheduling.id) {
        let sequence = call.sequence;
        if parent.is_none() {
            let assignment = runmat_execution::ProgramExecutionAssignment {
                scope_id: scheduling.scope_id,
                pool_id: scheduling.task.pool_id,
                task_id: scheduling.task_id,
                attempt_id: scheduling.id,
                worker_id: scheduling.worker_id,
                backend: runmat_execution::PoolBackend::Remote,
                resources: scheduling.resource_assignment.clone(),
            };
            match crate::fea_proxy::Parent::new(
                provider.clone(),
                assignment,
                scheduling.task.resources.clone(),
                scheduling.task.deadline_unix_millis,
                0,
            ) {
                Ok(created) => *parent = Some(created),
                Err(_) => {
                    channel.complete_fea_call(
                        scheduling.id,
                        sequence,
                        crate::fea_proxy::Reply::Denied {
                            denied: runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LicenseInvalid,
                        },
                    ).await?;
                    continue;
                }
            }
        }
        let owner = std::sync::Arc::clone(parent.as_ref().ok_or_else(|| {
            NativeExecutionError::Protocol("remote FEA parent was not installed".into())
        })?);
        let reply = tokio::task::spawn_blocking(move || owner.call(call))
            .await
            .map_err(|error| {
                NativeExecutionError::Protocol(format!("remote FEA parent stopped: {error}"))
            })?;
        channel
            .complete_fea_call(scheduling.id, sequence, reply)
            .await?;
    }
    Ok(())
}

async fn forward_collectives(
    channel: &dyn RemoteWorkerChannel,
    attempt_id: AttemptId,
    collectives: Option<&RemoteCollectiveAssignment>,
) -> NativeExecutionResult<()> {
    let requests = channel.drain_collective_requests(attempt_id);
    if requests.is_empty() {
        return Ok(());
    }
    let assignment = collectives.ok_or_else(|| {
        NativeExecutionError::Protocol(
            "non-SPMD remote attempt emitted a collective request".into(),
        )
    })?;
    for request in requests {
        if request.context != assignment.context {
            return Err(NativeExecutionError::Protocol(
                "remote collective request differs from its scheduled rank assignment".into(),
            ));
        }
        let broker = std::sync::Arc::clone(&assignment.broker);
        let submitted = request.clone();
        let result = tokio::task::spawn_blocking(move || broker.execute_remote(submitted))
            .await
            .map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "remote collective coordinator stopped: {error}"
                ))
            })?;
        let result = match result {
            Ok(response) => crate::protocol::CollectiveProcessResult::Completed { response },
            Err(message) => crate::protocol::CollectiveProcessResult::Failed { message },
        };
        channel
            .complete_collective(attempt_id, &request, result)
            .await?;
    }
    Ok(())
}
