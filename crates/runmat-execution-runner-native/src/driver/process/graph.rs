//! Durable Graph attempts use the same staged launch and process-tree owner as local tasks.
use std::sync::{atomic::Ordering, Mutex};

use super::owned::{OwnedProcessControl, ProcessOwner};
use super::*;
use crate::driver::GraphAttemptAuthority;
use crate::process_provider::ProcessControl;

struct GraphControl<'a> {
    original: &'a dyn ProcessControl,
    authority: &'a GraphAttemptAuthority,
}
impl OwnedProcessControl for GraphControl<'_> {
    fn is_cancelled(&self) -> bool {
        self.original.cancellation().load(Ordering::Acquire)
    }
    fn cancellation_reason(&self) -> Option<runmat_execution::CancellationReason> {
        self.original.cancellation_reason()
    }
    fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        self.original.record_progress(progress)
    }
    fn graph_authority(&self) -> Option<&GraphAttemptAuthority> {
        Some(self.authority)
    }
}

#[derive(Default)]
struct Quarantine(Mutex<Vec<GraphProcessQuarantine>>);
impl Drop for Quarantine {
    fn drop(&mut self) {
        let entries = self.0.get_mut().expect("Graph quarantine poisoned");
        retry_unverified_quarantine(entries);
        persist_unverified_quarantine(entries);
    }
}

pub(crate) async fn execute_attempt(
    provider: &crate::process_provider::NativeProcessProvider<'_>,
    request: &AttemptRequest,
    original: &runmat_execution_runner::TaskSubmission,
    graph: runmat_execution_artifact::GraphExecutionRequest,
    control: &dyn ProcessControl,
) -> LocalAttemptObservation {
    let started = Instant::now();
    let meters = TransportMeters::default();
    let mut termination = AttemptExecutionTermination::NotStarted;
    let mut console = None;
    let mut retirement = LocalRetirementEvidence::NotStarted;
    let quarantine = Quarantine::default();
    let result = if control.cancellation().load(Ordering::Acquire) {
        // No staging or child ownership exists for this attempt yet.
        Err(TransferFailure::Cancelled)
    } else {
        match prepare(provider, request, original, graph) {
            Ok((launch, authority)) => {
                let owner = ProcessOwner {
                    config: provider.config,
                    artifacts: provider.artifacts,
                    objects: provider.objects,
                    graph_quarantine: &quarantine.0,
                };
                let scoped = GraphControl {
                    original: control,
                    authority: &authority,
                };
                owned::run_process(
                    &owner,
                    launch,
                    &scoped,
                    &meters,
                    &mut termination,
                    &mut console,
                    &mut retirement,
                )
                .await
            }
            Err(error) => Err(error),
        }
    };
    // Keep the existing retirement result authoritative even if the control
    // channel also failed; transport failure is never proof of cancellation.
    let result = if let Some(message) = control.failure() {
        match result {
            Err(TransferFailure::RetirementUnverified(message)) => {
                Err(TransferFailure::RetirementUnverified(message))
            }
            _ => Err(TransferFailure::Infrastructure(message)),
        }
    } else {
        result
    };
    drop(quarantine);
    observe(
        request,
        started,
        result,
        meters,
        termination,
        console,
        retirement,
    )
}

fn prepare(
    provider: &crate::process_provider::NativeProcessProvider<'_>,
    request: &AttemptRequest,
    original: &runmat_execution_runner::TaskSubmission,
    graph: runmat_execution_artifact::GraphExecutionRequest,
) -> Result<(WorkerLaunchRequest, GraphAttemptAuthority), TransferFailure> {
    let invalid = |error: String| TransferFailure::Infrastructure(error);
    request
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if request.task != original.request {
        return Err(invalid(
            "durable Graph assignment differs from its original task".into(),
        ));
    }
    crate::graph_submission::validate_original(&graph, original)
        .map_err(|error| invalid(error.to_string()))?;
    let admitted = graph
        .into_validated()
        .map_err(|error| invalid(error.to_string()))?;
    let graph = admitted.request();
    #[derive(serde::Serialize)]
    struct BorrowedStoredGraph<'a> {
        request: &'a runmat_execution_artifact::GraphExecutionRequest,
    }
    let stored = provider
        .artifacts
        .put_streamed_graph(
            &BorrowedStoredGraph { request: graph },
            runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES,
        )
        .map_err(|error| invalid(error.to_string()))?;
    crate::graph_submission::validate_floor(graph, original, stored.length)
        .map_err(|error| invalid(error.to_string()))?;
    crate::graph_submission::validate_scratch(&admitted, original)
        .map_err(|error| invalid(error.to_string()))?;
    let authority = GraphAttemptAuthority {
        request_artifact_id: crate::graph_submission::primary_artifact(graph),
        artifact_id: stored.id,
        stored_digest: stored.digest,
        stored_length: stored.length,
        context: graph.context.clone(),
        requested_outputs: graph.requested_outputs,
        callable: graph
            .scheduler_callable()
            .map_err(|error| invalid(error.to_string()))?,
        response: admitted
            .response_authority()
            .map_err(|error| invalid(error.to_string()))?,
    };
    let assignment = runmat_execution::ProgramExecutionAssignment {
        scope_id: request.scope_id,
        pool_id: request.task.pool_id,
        task_id: request.task_id,
        attempt_id: request.id,
        worker_id: request.worker_id,
        backend: runmat_execution::PoolBackend::LocalProcesses,
        resources: request.resource_assignment.clone(),
    };
    Ok((
        staged_graph_launch(request, &authority, assignment)?,
        authority,
    ))
}
