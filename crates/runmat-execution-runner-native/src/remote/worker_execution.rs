use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptSuccess};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingProgress};
use runmat_meshing_execution::{MeshingProgressSink, MeshingStageKernel};

use super::object_transfer::RemoteObjectStore;

#[derive(Default)]
pub(super) struct AttemptCancellation(Arc<AtomicBool>);

impl AttemptCancellation {
    pub(super) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(super) fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
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
    progress_sender: tokio::sync::mpsc::Sender<crate::ProgramProgress>,
) -> ProgramExecutionResponse {
    if program.artifact.form == ExecutableForm::AnalysisWorkload {
        drop(progress_sender);
        return tokio::task::spawn_blocking(move || {
            let _guard = install_analysis_cancellation(&cancellation);
            crate::execute_analysis_program_request(
                &program,
                &mut objects,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
            )
        })
        .await
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    if program.artifact.form == ExecutableForm::MeshingWorkload {
        let Some(host) = meshing_host else {
            return failure("remote worker has no meshing host capability");
        };
        if program.recipe.execution_mode == crate::EXACT_MESHING_ROOT_EXECUTION_MODE {
            drop(progress_sender);
            return tokio::task::spawn_blocking(move || {
                crate::exact_meshing_root::execute_exact_meshing_root_admission(
                    &program,
                    &mut objects,
                    cancellation.flag(),
                    host.limits,
                )
            })
            .await
            .unwrap_or_else(|error| failure(&error.to_string()));
        }
        return tokio::task::spawn_blocking(move || {
            let progress_error = Arc::new(Mutex::new(None));
            let mut progress = ChannelProgress {
                sender: progress_sender,
                error: Arc::clone(&progress_error),
            };
            let response = crate::execute_meshing_program_request(
                &program,
                &mut objects,
                host.kernel.as_ref(),
                cancellation.as_ref(),
                &mut progress,
                host.limits,
            );
            let progress_error = progress_error
                .lock()
                .expect("remote progress error poisoned")
                .take();
            match progress_error {
                Some(error) => failure(&error),
                None => response,
            }
        })
        .await
        .unwrap_or_else(|error| failure(&error.to_string()));
    }
    drop(progress_sender);
    crate::execute_host_program_request_with_project(
        program,
        project
            .as_deref()
            .map(crate::materialized_project::MaterializedProject::handoff),
    )
    .await
}

fn install_analysis_cancellation(
    cancellation: &AttemptCancellation,
) -> runmat_analysis_fea::FeaProgressContextGuard {
    let cancellation_flag = cancellation.flag();
    runmat_analysis_fea::replace_fea_progress_context(
        None,
        Some(Arc::new(move || cancellation_flag.load(Ordering::Acquire))),
    )
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

struct ChannelProgress {
    sender: tokio::sync::mpsc::Sender<crate::ProgramProgress>,
    error: Arc<Mutex<Option<String>>>,
}

impl MeshingProgressSink for ChannelProgress {
    fn record(&mut self, progress: &MeshingProgress) {
        let result = crate::meshing_host::encode_meshing_progress(progress).and_then(|progress| {
            self.sender
                .blocking_send(progress)
                .map_err(|_| "remote progress receiver closed".to_string())
        });
        if let Err(error) = result {
            *self.error.lock().expect("remote progress error poisoned") = Some(error);
        }
    }
}

fn failure(message: &str) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{install_analysis_cancellation, AttemptCancellation};

    #[test]
    fn remote_attempt_cancellation_reaches_analysis_solver_context() {
        let cancellation = AttemptCancellation::default();
        let _guard = install_analysis_cancellation(&cancellation);
        assert!(!runmat_analysis_fea::is_fea_cancelled());
        cancellation.cancel();
        assert!(runmat_analysis_fea::is_fea_cancelled());
    }
}
