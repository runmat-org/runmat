//! Admit a V3 unit before polling its selected body.

use super::*;
use std::task::{Context, Poll};

pin_project_lite::pin_project! {
    struct UnitContinuation<F, O> {
        #[pin]
        body: Option<F>,
        owners: Option<O>,
    }
}

impl<F, O> Future for UnitContinuation<F, O>
where
    F: Future<Output = ProgramExecutionResponse>,
{
    type Output = ProgramExecutionResponse;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        let response = this
            .body
            .as_mut()
            .as_pin_mut()
            .expect("unit continuation polled after completion")
            .poll(cx);
        if response.is_ready() {
            // Drop the completed body while its selected receipt and metadata
            // are still retained, as on cancellation via field drop order.
            this.body.set(None);
            this.owners.take();
        }
        response
    }
}

fn continuation<F, O>(body: F, owners: O) -> PreparedRequest
where
    F: Future<Output = ProgramExecutionResponse> + 'static,
    O: 'static,
{
    // Replace the existing unit branch box; the pin projection adds no owner
    // allocation and never moves a body out of its pinned storage to poll it.
    Box::pin(UnitContinuation {
        body: Some(body),
        owners: Some(owners),
    })
}

#[inline(never)]
pub(super) fn prepare(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    bundle: Option<Box<runmat_execution_artifact::ExecutionBundle>>,
) -> Result<PreparedRequest, ProgramPreparationError> {
    let admitted = unit_admission::admit_executable_unit(&request.artifact.executable_bytes)
        .map_err(|error| ProgramPreparationError {
            message: error.worker_message(),
        })?;
    // Resolve only the exact admitted entrypoint, not a structured region that
    // happens to share its numeric ordinal.
    if !matches!(
        (&request.callable, admitted.envelope().manifest.identity.entrypoint_function),
        (ProgramCallable::Semantic { function, .. }, Some(entrypoint)) if *function == entrypoint
    ) {
        return Err(ProgramPreparationError {
            message: "worker rejected executable callable outside its admitted entrypoint".into(),
        });
    }
    let entrypoint_kind = admitted.envelope().manifest.identity.entrypoint_kind;
    let (bound, receipt) = admitted
        .bind_runtime_sources_with_bundle(
            runtime,
            bundle.map(|bundle| *bundle),
            runmat_execution_artifact::archive::ArchiveLimits::default(),
        )
        .map_err(|error| ProgramPreparationError {
            message: error.worker_message(),
        })?;
    let selected_function = bound.entrypoint_function();
    let (bytecode, registry, layout, runtime) = bound.into_parts();

    match entrypoint_kind {
        runmat_execution::ExecutableEntrypointKind::Script => Ok(continuation(
            execute_unit_script(bytecode, request.requested_outputs, runtime),
            (request, registry, layout, receipt),
        )),
        runmat_execution::ExecutableEntrypointKind::Function => {
            // The local registry key comes from admission, not the wire ordinal.
            let selected_function = selected_function.ok_or_else(|| ProgramPreparationError {
                message: "worker rejected function entrypoint without a selected body".into(),
            })?;
            Ok(continuation(
                execute_function_request(request, registry, runtime, Some(selected_function)),
                (bytecode, layout, receipt),
            ))
        }
        runmat_execution::ExecutableEntrypointKind::DeclarationOnly => {
            Err(ProgramPreparationError {
                message: "worker rejected declaration-only executable as a callable".into(),
            })
        }
    }
}
