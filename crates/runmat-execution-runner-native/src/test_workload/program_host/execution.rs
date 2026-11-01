use super::{HostResponseFuture, HostUnitRequest};
use runmat_execution_artifact::{ExecutionBundle, ProgramExecutionResponse};

#[derive(Debug)]
pub(super) enum ForegroundFailure {
    Control(Box<runmat_runtime::RuntimeError>),
    Original(Box<ProgramExecutionResponse>),
}

impl From<runmat_runtime::RuntimeError> for ForegroundFailure {
    fn from(error: runmat_runtime::RuntimeError) -> Self {
        Self::Control(Box::new(error))
    }
}

/// Admit the selected host/graphics driver before embedding it in the generic
/// local-task supervisor. Its concrete state has one quoted heap owner rather
/// than being copied through each foreground poll frame.
#[inline(never)]
pub(super) fn admitted_foreground_execution(
    runtime: &runmat_runtime::context::RuntimeContext,
    execution: HostResponseFuture<'static>,
    callbacks: Option<runmat_runtime::context::RuntimeGraphicsCallbackQueue>,
) -> Result<
    runmat_runtime::execution::RuntimeContinuationFuture<
        'static,
        ProgramExecutionResponse,
        ForegroundFailure,
    >,
    ForegroundFailure,
> {
    runmat_runtime::execution::admit_native_hostcall_with_contextual_error(
        runtime,
        move || async move {
            let response = match callbacks {
                Some(queue) => queue.drive_execution_with_host_callbacks(execution).await?,
                None => execution.await,
            };
            // Wire-encoded originating failures are errors, not successful output
            // publication. Preserve them if cancellation arrives on the final poll.
            match response {
                ProgramExecutionResponse::Failure { .. }
                | ProgramExecutionResponse::RuntimeFailure { .. } => {
                    Err(ForegroundFailure::Original(Box::new(response)))
                }
                ProgramExecutionResponse::Success { .. }
                | ProgramExecutionResponse::GraphSuccess { .. }
                | ProgramExecutionResponse::ExternalizedSuccess { .. }
                | ProgramExecutionResponse::SpmdSuccess { .. } => Ok(response),
            }
        },
    )
}

/// Construct the one selected interpreter future off the live host poll frame.
/// Pure Direct/Graph never enter this foreign/collective phase.
#[inline(never)]
pub(super) fn boxed_unit_execution(
    request: HostUnitRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    selected_bundle: Option<Box<ExecutionBundle>>,
    graph_failure_sources: runmat_vm::GraphFailureSourceRecorder,
) -> HostResponseFuture<'static> {
    match request {
        HostUnitRequest::Direct(request) => Box::pin(async move {
            if let Some(bundle) = selected_bundle {
                runmat_vm::execute_program_request_with_context_and_bundle(
                    *request, runtime, bundle,
                )
                .await
            } else {
                runmat_vm::execute_program_request_with_context(*request, runtime).await
            }
        }),
        HostUnitRequest::Graph(request) => {
            Box::pin(runmat_vm::execute_graph_request_with_context_recording(
                *request,
                runtime,
                graph_failure_sources,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn originating_wire_failures_survive_cancellation_on_the_final_poll() {
        let base = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        for original in [
            ProgramExecutionResponse::Failure {
                message: "originating worker failure".into(),
            },
            runmat_vm::runtime_failure_response(
                runmat_runtime::build_runtime_error("originating runtime failure")
                    .with_identifier("RunMat:test:ForegroundOrigin")
                    .build(),
                &base,
            ),
        ] {
            let source = runmat_execution::cancellation::CancellationSource::new();
            let runtime = base.clone().with_cancellation_token(source.token());
            let expected = original.clone();
            let execution = Box::pin(async move {
                source.cancel(runmat_execution::CancellationReason::User);
                original
            });
            let future = admitted_foreground_execution(&runtime, execution, None).unwrap();
            match futures::executor::block_on(future) {
                Err(ForegroundFailure::Original(response)) => assert_eq!(*response, expected),
                other => panic!("originating failure was replaced: {other:?}"),
            }
        }
    }
}
