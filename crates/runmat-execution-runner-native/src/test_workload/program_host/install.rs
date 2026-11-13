use std::sync::Arc;

use super::SelectedDirectRoute;
use runmat_execution_artifact::ProgramExecutionResponse;

/// Only an admitted foreign Direct route pays for the install-phase future.
/// Its large Core installer state is not embedded in the host continuation's
/// live poll frame or in the ordinary pure Direct/Graph route.
#[inline(never)]
pub(super) fn boxed_selected_install<'a>(
    session: &'a mut runmat_core::RunMatSession,
    selected: &'a SelectedDirectRoute<'_>,
    interop: &'a runmat_types::InteropManifest,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ProgramExecutionResponse>> + 'a>>
{
    Box::pin(async move {
        match (&selected.prepared, selected.handoff) {
            (Some(prepared), Some(handoff)) => {
                session
                    .install_selected_project_handoff(
                        handoff.clone(),
                        Arc::clone(prepared),
                        interop,
                        &selected.view,
                    )
                    .await
                    .map_err(|error| ProgramExecutionResponse::Failure {
                        message: format!("worker could not install exact Direct handoff: {error}"),
                    })?;
            }
            (None, None) => {
                session
                    .install_selected_interop_without_handoff(interop, &selected.view)
                    .await
                    .map_err(|error| ProgramExecutionResponse::Failure {
                        message: format!("worker rejected exact Direct interop: {error}"),
                    })?;
            }
            _ => {
                return Err(ProgramExecutionResponse::Failure {
                    message: "selected Direct handoff lost its prepared receipt".into(),
                });
            }
        }
        Ok(())
    })
}
