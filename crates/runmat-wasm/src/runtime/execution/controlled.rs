//! One selected program invocation and its exact Runtime retirement boundary.
//! This is not a worker scheduler and does not reinterpret physical-exit APIs.

use std::num::NonZeroU64;
use std::rc::Rc;

use runmat_execution::{CancellationToken, ProgramExecutionAssignment, ProgramRuntimeFailure};
use runmat_execution_artifact::{
    ProgramExecutionResponse, ProgramTaskRetirement, ProgramTaskTerminal,
    PROGRAM_TASK_TERMINAL_SCHEMA_VERSION,
};
use runmat_runtime::context::{BrowserLocalTaskSupervisor, RuntimeContext, RuntimeLifecycleHost};
use runmat_runtime::execution::{CpuCooperationPolicy, RuntimeExecutionService};

use super::admission::validate_selected_worker_admission;
use super::request::BrowserWorkerRequest;
use super::task_turn::BrowserTaskTurn;

mod factory;
pub(crate) use factory::inventory;

pub(crate) fn assignment(
    request: &BrowserWorkerRequest,
) -> Result<&ProgramExecutionAssignment, String> {
    let assignment = match request {
        BrowserWorkerRequest::Direct(request) => request.assignment.as_ref(),
        BrowserWorkerRequest::Graph(request) => request.assignment.as_ref(),
        BrowserWorkerRequest::OwnedBuiltin(_) => {
            return Err("owned builtin work requires its existing physical-exit protocol".into());
        }
    }
    .ok_or_else(|| "selected worker invocation has no scheduler assignment".to_owned())?;
    assignment.validate().map_err(|error| error.to_string())?;
    if assignment.backend != runmat_execution::PoolBackend::BrowserWorkers {
        return Err("selected worker invocation requires a browser-worker assignment".into());
    }
    Ok(assignment)
}

fn task_turn() -> runmat_runtime::context::RuntimeServiceFuture<
    Result<(), runmat_runtime::execution::ExecutionServiceError>,
> {
    Box::pin(async { BrowserTaskTurn::new()?.await })
}

fn admit(request: &BrowserWorkerRequest) -> Result<(), String> {
    // This invocation owner is for a fresh CPU worker. A separately initialized
    // provider requires its own device-settlement owner, not a Boolean claim.
    if runmat_accelerate_api::next_registered_provider(None).is_some() {
        return Err("selected worker retirement requires an unprovisioned CPU worker".into());
    }
    if runmat_filesystem::current_provider_supports_read() {
        return Err(
            "selected worker retirement requires its unprovisioned filesystem profile".into(),
        );
    }
    validate_selected_worker_admission(request, &inventory()?)?;
    match request {
        BrowserWorkerRequest::Direct(_) => Ok(()),
        BrowserWorkerRequest::Graph(request) => {
            let requirements = request
                .selected_unit_host_requirements()
                .map_err(|error| error.to_string())?;
            if requirements.iter().any(|requirement| {
                !requirement.interop.foreign_types.is_empty()
                    || !requirement.interop.adapters.is_empty()
                    || !requirement.interop.adapter_contracts.is_empty()
            }) {
                return Err("graph interop requires an explicitly provisioned worker host".into());
            }
            Ok(())
        }
        BrowserWorkerRequest::OwnedBuiltin(_) => {
            Err("owned builtin work requires its existing physical-exit protocol".into())
        }
    }
}

/// The request is moved into this invocation exactly once. The controller
/// retains only its actual assignment and cancellation source, not input roots.
pub(crate) async fn execute(
    request: BrowserWorkerRequest,
    assignment: ProgramExecutionAssignment,
    cancellation: CancellationToken,
) -> ProgramTaskTerminal {
    if let Err(message) = admit(&request) {
        drop(request);
        return terminal(
            assignment,
            ProgramExecutionResponse::Failure { message },
            ProgramTaskRetirement::Retired,
        );
    }
    let execution = Rc::new(RuntimeExecutionService::new().with_cpu_task_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(16_384).expect("nonzero quantum")),
        task_turn,
    ));
    let runtime = RuntimeContext::with_cancellation(execution, cancellation);
    runtime.enable_worker_invocation_resources();
    let retired = runtime.worker_invocation_retirement_check();
    let workspace = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let workspace_service = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
        workspace.clone(),
    ));
    let supervisor = Rc::new(BrowserLocalTaskSupervisor::new());
    let ports = factory::ports()
        // No console transport is provisioned in this isolated CPU worker.
        // Do not retain fallback output or inherit interactive/UI services.
        .with_host(Rc::new(runmat_runtime::context::NonCapturingPortableHost))
        .with_workspace(workspace_service.clone())
        .with_base_workspace(workspace_service)
        .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
    let runtime = runtime.with_service_ports(ports);
    let job = match &request {
        BrowserWorkerRequest::Direct(request) => request.job_id,
        BrowserWorkerRequest::Graph(request) => request.job_id,
        BrowserWorkerRequest::OwnedBuiltin(_) => unreachable!("admitted selected request"),
    };
    // Retain original authority through cleanup and resource retirement too,
    // not merely the nested VM's execution/output-encoding extent.
    let assignment_scope = runtime.enter_execution_assignment(Some(assignment.clone()));
    let job_scope = runtime.enter_execution_job(job);
    let sources = runmat_vm::GraphFailureSourceRecorder::new();
    let mut cleanup = CleanupDisposition::default();
    let response = match RuntimeLifecycleHost::new(&runtime, supervisor.clone()) {
        Ok(host) => {
            let response = match request {
                BrowserWorkerRequest::Direct(request) => {
                    runmat_vm::execute_program_request_with_context(request, runtime.clone()).await
                }
                BrowserWorkerRequest::Graph(request) => {
                    runmat_vm::execute_graph_request_with_context_recording(
                        request,
                        runtime.clone(),
                        sources.clone(),
                    )
                    .await
                }
                BrowserWorkerRequest::OwnedBuiltin(_) => unreachable!("admitted selected request"),
            };
            if let Err(error) = host.retire_workspace_and_drain(&runtime).await {
                cleanup.record(error, &runtime, &sources);
            }
            // Cancellation of a child is not its drop fence. In particular a
            // currently polling child cannot certify its own frame retirement.
            host.force_close_and_retire().await;
            drop(host);
            response
        }
        Err(error) => {
            drop(request);
            supervisor.shutdown_and_retire().await;
            runmat_vm::runtime_failure_response(error, &runtime)
        }
    };
    if let Err(error) = workspace.close() {
        cleanup.record(
            runmat_runtime::workspace::base_workspace_error_to_runtime(error),
            &runtime,
            &sources,
        );
    }
    runtime.retire_worker_invocation_resources();
    drop(sources);
    drop(supervisor);
    drop(workspace);
    drop(job_scope);
    drop(assignment_scope);
    drop(runtime);
    if runmat_accelerate_api::next_registered_provider(None).is_some() {
        cleanup.note("worker provider retirement is not verified".into());
    }
    // Response payloads are wire values, not Runtime/GC roots. Collect only
    // after all input/source/context owners and actual child frames are gone.
    if let Err(error) = RuntimeContext::collect_retired_worker_values() {
        cleanup.note(format!("worker GC retirement failed: {error}"));
    }
    if !retired() {
        cleanup.note("worker invocation still has live context or execution owners".into());
    }
    terminal(assignment, response, cleanup.finish())
}

fn terminal(
    assignment: ProgramExecutionAssignment,
    response: ProgramExecutionResponse,
    retirement: ProgramTaskRetirement,
) -> ProgramTaskTerminal {
    ProgramTaskTerminal {
        schema_version: PROGRAM_TASK_TERMINAL_SCHEMA_VERSION,
        assignment,
        response: Some(response),
        retirement,
    }
}

#[derive(Default)]
struct CleanupDisposition {
    diagnostic: Option<String>,
    failure: Option<ProgramRuntimeFailure>,
}

impl CleanupDisposition {
    fn note(&mut self, diagnostic: String) {
        if let Some(existing) = &mut self.diagnostic {
            existing.push_str("; ");
            existing.push_str(&diagnostic);
        } else {
            self.diagnostic = Some(diagnostic);
        }
    }

    fn record(
        &mut self,
        error: runmat_runtime::RuntimeError,
        runtime: &RuntimeContext,
        sources: &runmat_vm::GraphFailureSourceRecorder,
    ) {
        self.note(error.to_string());
        if self.failure.is_none() {
            self.failure = match sources.encode_runtime_failure(error, runtime) {
                ProgramExecutionResponse::RuntimeFailure { failure } => Some(failure),
                _ => None,
            };
        }
    }

    fn finish(self) -> ProgramTaskRetirement {
        match self.diagnostic {
            None => ProgramTaskRetirement::Retired,
            Some(diagnostic) => ProgramTaskRetirement::Unverified {
                diagnostic,
                cleanup_failure: self.failure.map(Box::new),
            },
        }
    }
}
