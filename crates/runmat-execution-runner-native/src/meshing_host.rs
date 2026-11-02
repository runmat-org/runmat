//! Meshing-capable native worker seam; scheduling remains in the generic execution driver.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
use runmat_process_host::ipc::{read_payload_blocking, FrameLimits};

use crate::protocol::{ProgramProgress, WorkerProcessMessage, NATIVE_WORKER_MESSAGE_SCHEMA_V2};
use crate::{FilesystemObjectStore, NativeExecutionError, NativeExecutionResult};

const MESHING_PROGRESS_MEDIA_TYPE: &str = "application/vnd.runmat.meshing-progress+cbor";
const MESHING_PROGRESS_VALUE_SCHEMA: &str = "runmat.meshing.progress.v2";
const PROGRESS_CHANNEL_CAPACITY: usize = 256;
const PARENT_DISCONNECT_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

#[derive(Default)]
struct ParentConnectionCancellation(AtomicBool);

impl ParentConnectionCancellation {
    fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

impl MeshingCancellationSignal for ParentConnectionCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
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
    let control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(runmat_process_host::ProcessHostError::Io)?;
    let writer = ControlFrameWriter::new(control.writer, frame_limits)?;
    // The one-shot request is read off the async executor. Return the very same
    // private reader to the watchdog so parent EOF remains observable afterward.
    let (reader, payload) = tokio::task::spawn_blocking(move || {
        let mut reader = control.reader;
        let payload = read_payload_blocking(&mut reader, frame_limits);
        (reader, payload)
    })
    .await
    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let payload = payload?;
    let reserved: runmat_execution_artifact::ReservedProgramExecutionRequest =
        serde_json::from_slice(&payload)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    // The decoded request owns its fields. Keep no redundant wire buffer alive
    // across the asynchronous native stage and its numerical working storage.
    drop(payload);
    reserved
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let (request, resources) = reserved.into_parts();
    let cpu_threads = admitted_cpu_threads(resources.cpu_millicores, resources.memory_bytes)?;
    tracing::info!(target: "runmat::meshing::cpu", cpu_threads,
        admitted_cpu_millicores = resources.cpu_millicores,
        admitted_memory_bytes = resources.memory_bytes,
        pool_stack_bytes = if cpu_threads > 1 { cpu_threads as u64 * runmat_cpu_execution::WORKER_STACK_BYTES } else { 0 },
        "meshing worker CPU allocation");
    let cancellation = Arc::new(ParentConnectionCancellation::default());
    let parent_connection = Arc::clone(&cancellation);
    // The parent keeps the private request channel open until this one-shot
    // worker exits. Observe it on a detached thread, including during a
    // non-yielding native stage.
    let parent_watchdog = ParentDisconnectWatchdog::spawn(reader, parent_connection)?;
    let store = FilesystemObjectStore::open(object_store_root, limits.inventory.max_object_bytes)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let (progress_sender, mut progress_receiver) =
        tokio::sync::mpsc::channel(PROGRESS_CHANNEL_CAPACITY);
    let progress_error = Arc::new(Mutex::new(None));
    let sink_error = Arc::clone(&progress_error);
    let execution_cancellation = Arc::clone(&cancellation);
    let mut execution = tokio::task::spawn_blocking(move || {
        let mut store = store;
        let mut progress = ChannelProgress {
            sender: progress_sender,
            error: sink_error,
        };
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
    });
    let response = loop {
        tokio::select! {
            progress = progress_receiver.recv() => {
                if let Some(progress) = progress {
                    write_message(&writer, &WorkerProcessMessage::Progress { progress }, frame_limits).await?;
                }
            }
            response = &mut execution => {
                while let Ok(progress) = progress_receiver.try_recv() {
                    write_message(&writer, &WorkerProcessMessage::Progress { progress }, frame_limits).await?;
                }
                break response
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            }
        }
    };
    parent_watchdog.finish();
    if let Some(error) = progress_error
        .lock()
        .expect("meshing progress error poisoned")
        .take()
    {
        return Err(NativeExecutionError::Protocol(error));
    }
    write_message(
        &writer,
        &WorkerProcessMessage::Completed {
            response,
            console: None,
        },
        frame_limits,
    )
    .await?;
    Ok(())
}

fn cancel_on_parent_disconnect(
    mut reader: impl Read,
    cancellation: Arc<ParentConnectionCancellation>,
    finished: &AtomicBool,
) -> bool {
    let mut unexpected_input = [0_u8; 1];
    let _ = reader.read(&mut unexpected_input);
    if finished.load(Ordering::Acquire) {
        return false;
    }
    cancellation.cancel();
    true
}

struct ParentDisconnectWatchdog {
    finished: Arc<AtomicBool>,
}

impl ParentDisconnectWatchdog {
    fn spawn(
        reader: impl Read + Send + 'static,
        cancellation: Arc<ParentConnectionCancellation>,
    ) -> NativeExecutionResult<Self> {
        let finished = Arc::new(AtomicBool::new(false));
        let watched = Arc::clone(&finished);
        std::thread::Builder::new()
            .name("runmat-meshing-parent-watchdog".into())
            .spawn(move || {
                if !cancel_on_parent_disconnect(reader, cancellation, &watched) {
                    return;
                }
                std::thread::sleep(PARENT_DISCONNECT_SHUTDOWN_GRACE);
                if watched.load(Ordering::Acquire) {
                    return;
                }
                // A native worker has no useful independent lifetime: its object store, progress
                // stream, and terminal response all belong to the parent attempt. Cooperative
                // cancellation gets a short grace period for normal cleanup; a kernel that is
                // between cancellation checkpoints must not survive indefinitely as an orphan.
                std::process::exit(1);
            })
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        Ok(Self { finished })
    }

    fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }
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
    fn parent_disconnect_cancels_native_meshing_work() {
        let cancellation = Arc::new(ParentConnectionCancellation::default());
        let finished = AtomicBool::new(false);
        assert!(cancel_on_parent_disconnect(
            std::io::Cursor::new(Vec::<u8>::new()),
            Arc::clone(&cancellation),
            &finished,
        ));

        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn completed_worker_ignores_late_parent_disconnect() {
        let cancellation = Arc::new(ParentConnectionCancellation::default());
        let finished = AtomicBool::new(true);
        assert!(!cancel_on_parent_disconnect(
            std::io::Cursor::new(Vec::<u8>::new()),
            Arc::clone(&cancellation),
            &finished,
        ));
        assert!(!cancellation.is_cancelled());
    }
}
