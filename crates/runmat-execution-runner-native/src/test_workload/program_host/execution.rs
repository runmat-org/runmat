use super::{HostResponseFuture, HostUnitRequest};
use runmat_execution_artifact::ExecutionBundle;

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
                runmat_vm::execute_program_request_with_context_and_bundle(request, runtime, bundle)
                    .await
            } else {
                runmat_vm::execute_program_request_with_context(request, runtime).await
            }
        }),
        HostUnitRequest::Graph(request) => {
            Box::pin(runmat_vm::execute_graph_request_with_context_recording(
                request,
                runtime,
                graph_failure_sources,
            ))
        }
    }
}
