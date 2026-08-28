//! Meshing-capable native worker seam; scheduling remains in the generic execution driver.

use std::path::Path;
use std::sync::{Arc, Mutex};

use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingChunkPolicy, MeshingProgress,
    NeverCancelled,
};
use runmat_meshing_execution::{
    execute_serial_stage, MeshingHostResponse, MeshingHostWorkload, MeshingProgressSink,
    MeshingStageKernel,
};
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

use crate::protocol::{ProgramProgress, WorkerProcessMessage, NATIVE_WORKER_MESSAGE_SCHEMA_V1};
use crate::{NativeExecutionError, NativeExecutionResult, NativeObjectStore};

const MESHING_PROGRESS_MEDIA_TYPE: &str = "application/vnd.runmat.meshing-progress+cbor";
const MESHING_PROGRESS_VALUE_SCHEMA: &str = "runmat.meshing.progress.v2";
const PROGRESS_CHANNEL_CAPACITY: usize = 256;

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
    let host = match MeshingHostWorkload::from_program_request(request) {
        Ok(host) => host,
        Err(error) => return failure(error.to_string()),
    };
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
    let (mut reader, mut writer) = runmat_process_host::ipc::stdio::endpoint()?;
    let payload = read_payload(&mut reader, frame_limits).await?;
    let request: ProgramExecutionRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let store = NativeObjectStore::open(object_store_root, limits.inventory.max_object_bytes)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let (progress_sender, mut progress_receiver) =
        tokio::sync::mpsc::channel(PROGRESS_CHANNEL_CAPACITY);
    let progress_error = Arc::new(Mutex::new(None));
    let sink_error = Arc::clone(&progress_error);
    let mut execution = tokio::task::spawn_blocking(move || {
        let mut store = store;
        let mut progress = ChannelProgress {
            sender: progress_sender,
            error: sink_error,
        };
        execute_meshing_program_request(
            &request,
            &mut store,
            kernel.as_ref(),
            &NeverCancelled,
            &mut progress,
            limits,
        )
    });
    let response = loop {
        tokio::select! {
            progress = progress_receiver.recv() => {
                if let Some(progress) = progress {
                    write_message(
                        &mut writer,
                        &WorkerProcessMessage::Progress { progress },
                        frame_limits,
                    ).await?;
                }
            }
            response = &mut execution => {
                while let Ok(progress) = progress_receiver.try_recv() {
                    write_message(
                        &mut writer,
                        &WorkerProcessMessage::Progress { progress },
                        frame_limits,
                    ).await?;
                }
                break response
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            }
        }
    };
    if let Some(error) = progress_error
        .lock()
        .expect("meshing progress error poisoned")
        .take()
    {
        return Err(NativeExecutionError::Protocol(error));
    }
    write_message(
        &mut writer,
        &WorkerProcessMessage::Completed { response },
        frame_limits,
    )
    .await?;
    Ok(())
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
        schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V1,
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
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    message: &WorkerProcessMessage,
    limits: FrameLimits,
) -> NativeExecutionResult<()> {
    let payload = serde_json::to_vec(message)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    write_payload(writer, &payload, limits).await?;
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
