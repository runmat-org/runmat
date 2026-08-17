//! Meshing-capable native worker seam; scheduling remains in the generic execution driver.

use std::path::Path;

use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingChunkPolicyV2, NeverCancelled};
use runmat_meshing_execution::{
    execute_serial_stage, MeshingHostResponseV2, MeshingHostWorkloadV2, MeshingProgressSink,
    MeshingStageKernel, NoopMeshingProgress,
};
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

use crate::{NativeExecutionError, NativeExecutionResult, NativeObjectStore};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMeshingHostLimits {
    pub chunk_policy: MeshingChunkPolicyV2,
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
            chunk_policy: MeshingChunkPolicyV2 {
                maximum_chunk_bytes: 8 * 1024 * 1024,
                maximum_records_per_chunk: 65_536,
                maximum_total_encoded_bytes: 4 * 1024 * 1024 * 1024,
            },
            inventory: ObjectInventoryLimits::default(),
            max_message_bytes: 64 * 1024 * 1024,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute_meshing_program_request<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &impl MeshingStageKernel,
    cancellation: &dyn MeshingCancellationSignal,
    progress: &mut dyn MeshingProgressSink,
    limits: NativeMeshingHostLimits,
) -> ProgramExecutionResponse {
    if let Err(error) = limits.validate() {
        return failure(error.to_string());
    }
    let host = match MeshingHostWorkloadV2::from_program_request(request) {
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
        Ok(completed) => MeshingHostResponseV2::completed(&host, &completed)
            .map(|response| response.program_response()),
        Err(error) => match MeshingHostResponseV2::failed(&host, &error) {
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
    kernel: &impl MeshingStageKernel,
    object_store_root: &Path,
    limits: NativeMeshingHostLimits,
) -> NativeExecutionResult<()> {
    limits.validate()?;
    let frame_limits = FrameLimits {
        max_message_bytes: limits.max_message_bytes,
    };
    let (mut reader, mut writer) = runmat_process_host::ipc::stdio::endpoint();
    let payload = read_payload(&mut reader, frame_limits).await?;
    let request: ProgramExecutionRequest = serde_json::from_slice(&payload)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let mut store = NativeObjectStore::open(object_store_root, limits.inventory.max_object_bytes)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let mut progress = NoopMeshingProgress;
    let response = execute_meshing_program_request(
        &request,
        &mut store,
        kernel,
        &NeverCancelled,
        &mut progress,
        limits,
    );
    let payload = serde_json::to_vec(&response)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    write_payload(&mut writer, &payload, frame_limits).await?;
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
