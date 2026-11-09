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
impl crate::process_output::OutputDrainControl for GraphControl<'_> {
    fn raw_output_changed(&self) {
        self.original.raw_output_changed();
    }

    fn stop_for_raw_output(&self, error: std::sync::Arc<crate::RawOutputError>) {
        self.original.stop_for_raw_output(error);
    }

    fn raw_output_failure(&self) -> Option<std::sync::Arc<crate::RawOutputError>> {
        self.original.raw_output_failure()
    }
}
impl OwnedProcessControl for GraphControl<'_> {
    fn control_allocation_owner(
        &self,
    ) -> Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner> {
        // This durable Graph caller has an original physical task reservation,
        // but no Runtime-managed allocation owner. Do not fabricate one from
        // that reservation or claim that bootstrap storage is Runtime-managed.
        None
    }

    fn begin_raw_output(
        &self,
        attempt: &AttemptRequest,
        materialized: u64,
    ) -> Result<
        std::sync::Mutex<crate::process_output::RawOutputJournal>,
        std::sync::Arc<crate::RawOutputError>,
    > {
        self.original.begin_raw_output(attempt, materialized)
    }
    fn is_cancelled(&self) -> bool {
        self.original.cancellation().load(Ordering::Acquire)
            || self.original.raw_output_failure().is_some()
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
    fn record_console(
        &self,
        record: &runmat_execution::console::ConsoleRecord,
    ) -> Result<(), String> {
        self.original.record_console(record)
    }
    fn finish_console(
        &self,
        receipt: &runmat_execution::console::ConsoleReceipt,
    ) -> Result<(), String> {
        self.original.finish_console(receipt)
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
    let mut evidence = ProcessAttemptEvidence::new();
    let quarantine = Quarantine::default();
    let result = if control.cancellation().load(Ordering::Acquire) {
        // No staging or child ownership exists for this attempt yet.
        Err(TransferFailure::Cancelled)
    } else {
        match prepare(provider, request, original, graph) {
            Ok((launch, authority)) => {
                let owner = ProcessOwner {
                    attempt: request,
                    config: provider.config,
                    artifacts: provider.artifacts,
                    objects: provider.objects,
                    graph_quarantine: &quarantine.0,
                };
                let scoped = GraphControl {
                    original: control,
                    authority: &authority,
                };
                owned::run_process(&owner, launch, &scoped, &mut evidence).await
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
    evidence.finish(request, result)
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
    let materialized_scratch_bytes =
        crate::attempt_scratch::materialized_bytes(admitted.selected_bundles())
            .map_err(|error| invalid(error.to_string()))?;
    let authority = GraphAttemptAuthority {
        request_artifact_id: crate::graph_submission::primary_artifact(graph),
        artifact_id: stored.id,
        stored_digest: stored.digest,
        stored_length: stored.length,
        context: graph.context.clone(),
        requested_outputs: graph.requested_outputs,
        invocation_controls: graph.invocation_controls.clone().unwrap_or_default(),
        materialized_scratch_bytes,
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
        staged_graph_launch(
            request,
            &authority,
            assignment,
            request
                .task
                .console
                .map(|controls| crate::console_source_subgrant(&request.task.resources, controls))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?,
        )?,
        authority,
    ))
}
