use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptSuccess};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingProgressV2};
use runmat_meshing_execution::{MeshingProgressSink, MeshingStageKernel};

use super::object_transfer::RemoteObjectStore;

#[derive(Default)]
pub(super) struct AttemptCancellation(AtomicBool);

impl AttemptCancellation {
    pub(super) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

impl MeshingCancellationSignal for AttemptCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
pub(super) struct RemoteMeshingHost {
    kernel: Arc<dyn MeshingStageKernel>,
    limits: crate::NativeMeshingHostLimits,
}

impl RemoteMeshingHost {
    pub(super) fn new(
        kernel: Arc<dyn MeshingStageKernel>,
        limits: crate::NativeMeshingHostLimits,
    ) -> Self {
        Self { kernel, limits }
    }
}

pub(super) async fn execute(
    program: ProgramExecutionRequest,
    project: Option<Arc<crate::materialized_project::MaterializedProject>>,
    meshing_host: Option<RemoteMeshingHost>,
    mut objects: RemoteObjectStore,
    cancellation: Arc<AttemptCancellation>,
) -> ProgramExecutionResponse {
    if program.artifact.form == ExecutableForm::MeshingWorkloadV2 {
        let Some(host) = meshing_host else {
            return failure("remote worker has no meshing host capability");
        };
        return tokio::task::spawn_blocking(move || {
            crate::execute_meshing_program_request(
                &program,
                &mut objects,
                host.kernel.as_ref(),
                cancellation.as_ref(),
                &mut NoProgress,
                host.limits,
            )
        })
        .await
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    crate::execute_host_program_request_with_project(
        program,
        project
            .as_deref()
            .map(crate::materialized_project::MaterializedProject::handoff),
    )
    .await
}

pub(super) fn report(response: ProgramExecutionResponse) -> AttemptReport {
    match response {
        ProgramExecutionResponse::Success { value } => AttemptReport::Succeeded {
            result: AttemptSuccess {
                outputs: vec![value],
                result_objects: Vec::new(),
            },
        },
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => AttemptReport::Succeeded {
            result: AttemptSuccess {
                outputs,
                result_objects,
            },
        },
        ProgramExecutionResponse::Failure { message } => AttemptReport::Failed {
            kind: AttemptFailureKind::Execution,
            message,
        },
    }
}

struct NoProgress;

impl MeshingProgressSink for NoProgress {
    fn record(&mut self, _progress: &MeshingProgressV2) {}
}

fn failure(message: &str) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}
