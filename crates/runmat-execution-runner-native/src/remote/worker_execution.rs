use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess};
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

pub(super) fn report(
    request: &AttemptRequest,
    response: ProgramExecutionResponse,
    wall_millis: u64,
) -> AttemptReport {
    let success = match response {
        ProgramExecutionResponse::Success { value } => {
            AttemptSuccess::new(request, vec![value], Vec::new(), wall_millis, 0, 0)
        }
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => {
            let egress_bytes = result_objects.iter().try_fold(0_u64, |total, reference| {
                total.checked_add(reference.encoded_length)
            });
            match egress_bytes {
                Some(egress_bytes) => AttemptSuccess::new(
                    request,
                    outputs,
                    result_objects,
                    wall_millis,
                    egress_bytes,
                    0,
                ),
                None => Err("remote result egress accounting overflowed".into()),
            }
        }
        ProgramExecutionResponse::Failure { message } => {
            return AttemptReport::Failed {
                kind: AttemptFailureKind::Execution,
                message,
            }
        }
    };
    match success {
        Ok(result) => AttemptReport::Succeeded { result },
        Err(message) => AttemptReport::Failed {
            kind: AttemptFailureKind::Rejected,
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
    use std::collections::BTreeSet;

    use runmat_execution::identity::{ArtifactId, AttemptId, WorkerId};
    use runmat_execution::resource::ResourceRequest;
    use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
    use runmat_execution::{Digest, ExecutionScopeId, OutputContract, PoolId, TaskId};
    use runmat_execution_artifact::ProgramExecutionResponse;
    use runmat_execution_runner::{AttemptReport, AttemptRequest};

    use super::{install_analysis_cancellation, report, AttemptCancellation};

    #[test]
    fn remote_attempt_cancellation_reaches_analysis_solver_context() {
        let cancellation = AttemptCancellation::default();
        let _guard = install_analysis_cancellation(&cancellation);
        assert!(!runmat_analysis_fea::is_fea_cancelled());
        cancellation.cancel();
        assert!(runmat_analysis_fea::is_fea_cancelled());
    }

    #[test]
    fn metered_attempt_report_round_trips_over_the_remote_json_contract() {
        let task_id = TaskId::derive(&[b"task"]);
        let request = AttemptRequest {
            id: AttemptId::derive(&[b"attempt"]),
            task_id,
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            ordinal: 0,
            driver_fence: 1,
            task: TaskRequest {
                id: task_id,
                scope_id: ExecutionScopeId::derive(&[b"scope"]),
                pool_id: PoolId::derive(&[b"pool"]),
                program_artifact_id: ArtifactId::derive(&[b"artifact"]),
                callable: Callable {
                    owner_identity: "test".into(),
                    qualified_name: "test".into(),
                    entrypoint_digest: Digest::sha256(b"entrypoint"),
                },
                inputs: Vec::new(),
                outputs: OutputContract {
                    requested_outputs: 1,
                },
                resources: ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 16 * 1024 * 1024,
                    scratch_bytes: 8 * 1024 * 1024,
                    max_wall_millis: 1_000,
                    max_artifact_bytes: 1024,
                    max_egress_bytes: 1024,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: BTreeSet::new(),
                },
                retry: RetryPolicy::Never,
                deadline_unix_millis: None,
            },
        };
        let report = report(
            &request,
            ProgramExecutionResponse::Success {
                value: runmat_execution::value::ValuePayload::Inline(Box::new(
                    runmat_execution::value::InlineValue::Null,
                )),
            },
            1,
        );
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: AttemptReport = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, report);
    }
}
