//! Meshing-capable native worker seam; scheduling remains in the generic execution driver.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingChunkPolicy, MeshingProgress,
};
use runmat_meshing_execution::{
    execute_serial_stage, MeshingHostResponse, MeshingHostWorkload, MeshingProgressSink,
    MeshingStageKernel,
};
use runmat_process_host::ipc::private_control::ControlFrameWriter;
use runmat_process_host::ipc::FrameLimits;

use crate::protocol::{ProgramProgress, WorkerProcessMessage, NATIVE_WORKER_MESSAGE_SCHEMA_V2};
use crate::{FilesystemObjectStore, NativeExecutionError, NativeExecutionResult};

const MESHING_PROGRESS_MEDIA_TYPE: &str = "application/vnd.runmat.meshing-progress+cbor";
const MESHING_PROGRESS_VALUE_SCHEMA: &str = "runmat.meshing.progress.v2";
const PROGRESS_CHANNEL_CAPACITY: usize = 256;

#[derive(Default)]
struct ParentConnectionCancellation {
    stopped: AtomicBool,
    lease: Option<runmat_execution::cancellation::CancellationToken>,
}

impl ParentConnectionCancellation {
    fn cancel(&self) {
        self.stopped.store(true, Ordering::Release);
    }
}

impl MeshingCancellationSignal for ParentConnectionCancellation {
    fn is_cancelled(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
            || self
                .lease
                .as_ref()
                .is_some_and(|token| token.is_cancelled())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMeshingHostLimits {
    pub chunk_policy: MeshingChunkPolicy,
    pub inventory: ObjectInventoryLimits,
    pub max_message_bytes: u32,
}

impl NativeMeshingHostLimits {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.chunk_policy
            .validate()
            .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
        if self.inventory.max_objects == 0
            || self.inventory.max_object_bytes == 0
            || self.inventory.max_total_bytes == 0
            || self.max_message_bytes == 0
        {
            return Err(NativeExecutionError::Configuration(
                "native meshing host limits must all be non-zero".into(),
            ));
        }
        Ok(())
    }
}

impl Default for NativeMeshingHostLimits {
    fn default() -> Self {
        Self {
            chunk_policy: MeshingChunkPolicy {
                maximum_chunk_bytes: 8 * 1024 * 1024,
                maximum_records_per_chunk: 65_536,
                maximum_total_encoded_bytes: 4 * 1024 * 1024 * 1024,
            },
            inventory: ObjectInventoryLimits::default(),
            max_message_bytes: 64 * 1024 * 1024,
        }
    }
}

pub fn execute_meshing_program_request<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &dyn MeshingStageKernel,
    cancellation: &dyn MeshingCancellationSignal,
    progress: &mut dyn MeshingProgressSink,
    limits: NativeMeshingHostLimits,
) -> ProgramExecutionResponse {
    if let Err(error) = limits.validate() {
        return failure(error.to_string());
    }
    let host = match MeshingHostWorkload::from_program_request(request, store) {
        Ok(host) => host,
        Err(error) => return failure(error.to_string()),
    };
    // Blocking worker threads have no ambient async span. Give their events a
    // process-local owner so OTLP retains budget and kernel diagnostics.
    let worker_span = tracing::info_span!(
        "meshing.worker_stage",
        cpu_threads = runmat_cpu_execution::current_threads(),
        stage = ?host.workload.stage,
        partition_kind = ?host.workload.partition.kind,
        partition_index = host.workload.partition.partition_index,
        partition_count = host.workload.partition.partition_count,
    );
    let _worker_stage = worker_span.enter();
    let response = match execute_serial_stage(
        request,
        store,
        kernel,
        cancellation,
        progress,
        limits.chunk_policy,
        limits.inventory,
    ) {
        Ok(completed) => MeshingHostResponse::completed(&host, &completed)
            .map(|response| response.program_response()),
        Err(error) => match MeshingHostResponse::failed(&host, &error) {
            Ok(Some(response)) => Ok(response.program_response()),
            Ok(None) => return failure(error.to_string()),
            Err(response_error) => return failure(response_error.to_string()),
        },
    };
    match response {
        Ok(response) => match response.validate_against(request) {
            Ok(()) => response,
            Err(error) => failure(error.to_string()),
        },
        Err(error) => failure(error.to_string()),
    }
}

pub async fn run_meshing_worker_stdio(
    kernel: Arc<dyn MeshingStageKernel>,
    object_store_root: &Path,
    limits: NativeMeshingHostLimits,
) -> NativeExecutionResult<()> {
    limits.validate()?;
    let frame_limits = FrameLimits {
        max_message_bytes: limits.max_message_bytes,
    };
    let bootstrap = crate::worker::bootstrap::PrivateWorkerBootstrap::connect(
        frame_limits,
        crate::worker::bootstrap::WorkerRole::Meshing,
    )?;
    let payload = match run_meshing_worker(&bootstrap, kernel, object_store_root, limits).await {
        Ok(response) => runmat_canonical_codec::encode_json_bounded(
            &WorkerProcessMessage::Completed {
                response,
                console: None,
                console_scopes: Vec::new(),
            },
            u64::from(frame_limits.max_message_bytes),
        )?,
        Err(error) => bootstrap.failure_payload(error)?,
    };
    bootstrap.publish(payload).await
}

async fn run_meshing_worker(
    bootstrap: &crate::worker::bootstrap::PrivateWorkerBootstrap,
    kernel: Arc<dyn MeshingStageKernel>,
    object_store_root: &Path,
    limits: NativeMeshingHostLimits,
) -> NativeExecutionResult<ProgramExecutionResponse> {
    use crate::worker::bootstrap::AdmittedWorkerLaunch;
    let AdmittedWorkerLaunch {
        request: launch,
        scope,
        host,
    } = bootstrap.take_launch()?;
    // Role validation checked the exact original reservation and refused
    // Graph/SPMD/non-meshing and unsupported console forms before effects.
    let reserved = match launch {
        crate::protocol::WorkerLaunchRequest::Reserved(reserved) => reserved,
        _ => {
            return Err(NativeExecutionError::Protocol(
                "meshing launch retained unsupported invocation controls".into(),
            ))
        }
    };
    let (request, resources) = reserved.into_parts();
    let cpu_threads = admitted_cpu_threads(resources.cpu_millicores, resources.memory_bytes)?;
    tracing::info!(target: "runmat::meshing::cpu", cpu_threads,
        admitted_cpu_millicores = resources.cpu_millicores,
        admitted_memory_bytes = resources.memory_bytes,
        pool_stack_bytes = if cpu_threads > 1 { cpu_threads as u64 * runmat_cpu_execution::WORKER_STACK_BYTES } else { 0 },
        "meshing worker CPU allocation");
    let scope_control = scope.as_ref().map(|scope| scope.control());
    let cancellation = Arc::new(ParentConnectionCancellation {
        stopped: AtomicBool::new(false),
        lease: scope.as_ref().map(|scope| scope.cancellation_token()),
    });
    let store = FilesystemObjectStore::open(object_store_root, limits.inventory.max_object_bytes)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let (progress_sender, mut progress_receiver) =
        tokio::sync::mpsc::channel(PROGRESS_CHANNEL_CAPACITY);
    let progress_error = Arc::new(Mutex::new(None));
    let sink_error = Arc::clone(&progress_error);
    let execution_cancellation = Arc::clone(&cancellation);
    // The original paid scope/provider is entered on the actual numerical
    // thread. Retain its owner until the blocking task physically returns.
    let mut execution = tokio::task::spawn_blocking(move || {
        let mut store = store;
        let mut progress = ChannelProgress {
            sender: progress_sender,
            error: sink_error,
        };
        let mut run = || {
            runmat_cpu_execution::with_cpu_pool_cancellable(
                cpu_threads,
                || execution_cancellation.is_cancelled(),
                || {
                    execute_meshing_program_request(
                        &request,
                        &mut store,
                        kernel.as_ref(),
                        execution_cancellation.as_ref(),
                        &mut progress,
                        limits,
                    )
                },
            )
            .unwrap_or_else(|error| failure(error.to_string()))
        };
        match scope {
            Some(scope) => {
                let response = scope
                    .run_checked_sync(run)
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()));
                (response, Some(scope))
            }
            None => (
                Ok(match host {
                    Some(host) => host.run_sync(run),
                    None => run(),
                }),
                None,
            ),
        }
    });
    let frame_limits = FrameLimits {
        max_message_bytes: limits.max_message_bytes,
    };
    let mut control_error = None;
    let mut progress_open = true;
    let mut control_open = true;
    let response = loop {
        tokio::select! {
            progress = progress_receiver.recv(), if progress_open => {
                match progress {
                    Some(progress) => {
                        if control_error.is_none() {
                            if let Err(error) = write_message(bootstrap.writer(),
                                &WorkerProcessMessage::Progress { progress }, frame_limits).await {
                                cancellation.cancel();
                                control_error = Some(error);
                            }
                        }
                    }
                    None => progress_open = false,
                }
            }
            // Lease renewals/denials bypass this receiver in the shared
            // observer. A meshing stage has no other valid trailing controls.
            control = bootstrap.read_control(), if control_open => {
                cancellation.cancel();
                control_open = false;
                control_error.get_or_insert_with(|| match control {
                    Ok(_) => NativeExecutionError::Protocol("unexpected control for a meshing attempt".into()),
                    Err(error) => error,
                });
            }
            response = &mut execution => {
                while let Ok(progress) = progress_receiver.try_recv() {
                    if control_error.is_none() {
                        if let Err(error) = write_message(bootstrap.writer(),
                            &WorkerProcessMessage::Progress { progress }, frame_limits).await {
                            cancellation.cancel();
                            control_error = Some(error);
                        }
                    }
                }
                break response.map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            }
        }
    };
    // No control/lease/progress failure may detach a still-running kernel.
    if let Some(error) = control_error {
        return Err(error);
    }
    if let Some(error) = progress_error
        .lock()
        .expect("meshing progress error poisoned")
        .take()
    {
        return Err(NativeExecutionError::Protocol(error));
    }
    let (response, scope) = response;
    let response = response?;
    if let Some(control) = scope_control {
        control
            .check()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    if let Some(scope) = scope {
        let outcome = if matches!(&response, ProgramExecutionResponse::Failure { .. }) {
            runmat_runtime::analysis::fea_run_admission::FeaRunOutcome::Failed
        } else {
            runmat_runtime::analysis::fea_run_admission::FeaRunOutcome::Completed
        };
        scope
            .settle(outcome)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    Ok(response)
}

struct ChannelProgress {
    sender: tokio::sync::mpsc::Sender<ProgramProgress>,
    error: Arc<Mutex<Option<String>>>,
}

impl MeshingProgressSink for ChannelProgress {
    fn record(&mut self, progress: &MeshingProgress) {
        let message = encode_meshing_progress(progress).and_then(|progress| {
            self.sender
                .blocking_send(progress)
                .map_err(|_| "native progress receiver closed".to_string())
        });
        if let Err(error) = message {
            *self.error.lock().expect("meshing progress error poisoned") = Some(error);
        }
    }
}

pub(crate) fn encode_meshing_progress(
    progress: &MeshingProgress,
) -> Result<ProgramProgress, String> {
    let encoded = ProgramProgress {
        schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V2,
        sequence: progress.sequence,
        media_type: MESHING_PROGRESS_MEDIA_TYPE.into(),
        value_schema: MESHING_PROGRESS_VALUE_SCHEMA.into(),
        payload: progress
            .canonical_encode()
            .map_err(|error| error.to_string())?,
    };
    encoded.validate()?;
    Ok(encoded)
}

async fn write_message(
    writer: &ControlFrameWriter,
    message: &WorkerProcessMessage,
    limits: FrameLimits,
) -> NativeExecutionResult<()> {
    let payload =
        runmat_canonical_codec::encode_json_bounded(message, u64::from(limits.max_message_bytes))?;
    writer
        .write_owned(payload)
        .await
        .map_err(|error| runmat_process_host::ProcessHostError::Io(std::io::Error::other(error)))?;
    Ok(())
}

fn failure(message: String) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: if message.is_empty() {
            "native meshing host failed without a diagnostic".into()
        } else {
            message
        },
    }
}

/// One CPU/stack contract shared by dispatch and worker admission.
pub(crate) fn admitted_cpu_threads(
    cpu_millicores: u32,
    memory_bytes: u64,
) -> NativeExecutionResult<usize> {
    let cpu_threads = usize::try_from(cpu_millicores / 1_000)
        .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
    if cpu_threads == 0
        || !cpu_millicores.is_multiple_of(1_000)
        || cpu_threads > runmat_cpu_execution::MAX_THREADS
        || memory_bytes < cpu_threads as u64 * runmat_cpu_execution::WORKER_STACK_BYTES
    {
        return Err(NativeExecutionError::Configuration(
            "meshing CPU reservation lacks a whole-thread/stack envelope".into(),
        ));
    }
    Ok(cpu_threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_control_cancels_native_meshing_work() {
        let cancellation = ParentConnectionCancellation::default();
        cancellation.cancel();
        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn inherited_lease_denial_reaches_native_kernel_cancellation() {
        let grant = crate::fea_delegation::test_grant();
        let scope = grant.install(&grant.assignment, &grant.resources).unwrap();
        let cancellation = ParentConnectionCancellation {
            stopped: AtomicBool::new(false),
            lease: Some(scope.cancellation_token()),
        };
        assert!(!cancellation.is_cancelled());
        scope.deny(runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LeaseExpired);
        assert!(cancellation.is_cancelled());
    }
}
