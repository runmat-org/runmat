//! Explicit in-process graphics attachment. Process, remote and lab entrypoints
//! intentionally call the existing unattached route; no wire grants are added.

use super::program_host::{execute_host_unit_request_with_graphics, HostUnitRequest};
use runmat_core::SessionGraphicsHost;
use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, ProgramExecutionRequest, ProgramExecutionResponse,
};

/// The caller owns the OS-main-thread NativeGraphicsHost and invokes this on
/// its local execution lane. Both pure VM and host-dependent programs receive
/// the same explicit attachment through the existing Core frame owner.
pub async fn execute_host_program_request_with_graphics(
    request: ProgramExecutionRequest,
    graphics: &SessionGraphicsHost,
) -> ProgramExecutionResponse {
    if request.artifact.form == ExecutableForm::TestAttemptV1 {
        return ProgramExecutionResponse::Failure {
            message:
                "native graphics attachment requires a program request, not a test-attempt artifact"
                    .into(),
        };
    }
    execute_host_unit_request_with_graphics(
        HostUnitRequest::Direct(Box::new(request)),
        None,
        None,
        None,
        graphics,
    )
    .await
}

pub async fn execute_host_graph_request_with_graphics(
    request: GraphExecutionRequest,
    graphics: &SessionGraphicsHost,
) -> ProgramExecutionResponse {
    if request.selected_bundle_archives.iter().any(Option::is_some) {
        return ProgramExecutionResponse::Failure { message: "archive-bearing Graph interop requires an isolated attempt worker with exact selected-unit materialization".into() };
    }
    execute_host_unit_request_with_graphics(
        HostUnitRequest::Graph(Box::new(request)),
        None,
        None,
        None,
        graphics,
    )
    .await
}
