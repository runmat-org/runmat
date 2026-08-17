use std::path::PathBuf;

use base64::Engine as _;
use runmat_execution::identity::{PoolId, WorkerId};
use runmat_execution::resource::{AcceleratorRequest, ResourceInventory};
use runmat_execution_artifact::encryption::{decode_run_key_envelope, PortableExecutionEncryption};
use runmat_execution_runner::WorkerSpec;
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;
use sha2::{Digest as _, Sha256};

use super::worker_entry::{run_remote_worker_relay_cached, RemoteWorkerRelayRequest};
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WorkerResources {
    cpu_millicores: u32,
    memory_bytes: u64,
    scratch_bytes: u64,
    accelerator_count: u16,
    accelerator_class: Option<String>,
    accelerator_memory_bytes: u64,
    maximum_wall_millis: u64,
}

pub async fn run_remote_worker_from_env() -> NativeExecutionResult<()> {
    let run_id = required("RUNMAT_EXECUTION_RUN_ID")?;
    let allocation_id = required("RUNMAT_EXECUTION_ALLOCATION_ID")?;
    let allocation_fence = parse_u64("RUNMAT_EXECUTION_ALLOCATION_FENCING_TOKEN")?;
    let driver_fence = parse_u64("RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN")?;
    let endpoint_fingerprint = required("RUNMAT_EXECUTION_ENDPOINT_FINGERPRINT")?;
    let material_path = PathBuf::from(required("RUNMAT_EXECUTION_ENDPOINT_IDENTITY_FILE")?);
    if !material_path.is_absolute() || !material_path.is_file() {
        return Err(invalid("worker endpoint identity file is unavailable"));
    }
    let material: EndpointIdentityMaterial =
        serde_json::from_slice(&std::fs::read(material_path).map_err(protocol)?)
            .map_err(protocol)?;
    if material.evidence.run_identity != run_id
        || material.evidence.allocation_lease_id != allocation_id
        || material.evidence.fencing_token != allocation_fence
        || material.evidence.recipient.fingerprint != endpoint_fingerprint
    {
        return Err(invalid("worker endpoint identity does not match its lease"));
    }
    let (_, private_key) = material.recipient_private_key().map_err(protocol)?;
    let envelope_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(required("RUNMAT_EXECUTION_RUN_KEY_ENVELOPE")?)
        .map_err(protocol)?;
    let envelope = decode_run_key_envelope(&envelope_bytes, 64 * 1024).map_err(protocol)?;
    let run_key = PortableExecutionEncryption
        .open_run_key(&private_key, &envelope, &endpoint_fingerprint, &run_id, 1)
        .map_err(protocol)?;
    let resources: WorkerResources =
        serde_json::from_str(&required("RUNMAT_EXECUTION_WORKER_RESOURCES")?).map_err(protocol)?;
    if resources.maximum_wall_millis == 0 {
        return Err(invalid("worker maximum wall time is zero"));
    }
    let accelerators = match (resources.accelerator_count, resources.accelerator_class) {
        (0, _) => Vec::new(),
        (count, Some(class)) => vec![AcceleratorRequest {
            class,
            count,
            memory_bytes_each: resources.accelerator_memory_bytes / u64::from(count),
        }],
        _ => return Err(invalid("worker accelerator class is missing")),
    };
    let worker = WorkerSpec {
        id: WorkerId::derive(&[run_id.as_bytes(), allocation_id.as_bytes()]),
        pool_id: PoolId::derive(&[run_id.as_bytes(), b"remote-pool"]),
        resources: ResourceInventory {
            cpu_millicores: resources.cpu_millicores,
            memory_bytes: resources.memory_bytes,
            scratch_bytes: resources.scratch_bytes,
            accelerators,
            capabilities: Default::default(),
        },
    };
    let relay_url = worker_relay_url(
        &required("RUNMAT_EXECUTION_SERVER_URL")?,
        &required("RUNMAT_EXECUTION_WORKER_RELAY_PATH")?,
    )?;
    let relay_protocol = required("RUNMAT_EXECUTION_WORKER_RELAY_PROTOCOL")?;
    let relay_ticket = required("RUNMAT_EXECUTION_WORKER_RELAY_TICKET")?;
    let headers = [(
        "Sec-WebSocket-Protocol".to_string(),
        format!("{relay_protocol}, runmat-ticket.{relay_ticket}"),
    )];
    let bundle_cache = PathBuf::from(required("RUNMAT_EXECUTION_NODE_BUNDLE_CACHE")?);
    if !bundle_cache.is_absolute() {
        return Err(invalid("worker node bundle cache path must be absolute"));
    }
    run_remote_worker_relay_cached(
        RemoteWorkerRelayRequest {
            url: &relay_url,
            headers: &headers,
            run_identity: run_id.clone(),
            worker,
            driver_fence,
            session_id: session_id(&run_id, &allocation_id),
            run_key,
            limits: FrameLimits::default(),
        },
        Some(bundle_cache),
    )
    .await
}

fn worker_relay_url(server_url: &str, path: &str) -> NativeExecutionResult<String> {
    let base = server_url.trim_end_matches('/');
    let scheme = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        return Err(invalid("worker Server URL must use HTTP or HTTPS"));
    };
    if !path.starts_with('/') {
        return Err(invalid("worker relay path must be absolute"));
    }
    Ok(format!("{scheme}{path}"))
}

pub(crate) fn session_id(run_id: &str, allocation_id: &str) -> [u8; 16] {
    let digest = Sha256::digest(
        [
            b"runmat-worker-session-v1".as_slice(),
            run_id.as_bytes(),
            allocation_id.as_bytes(),
        ]
        .concat(),
    );
    let mut session = [0_u8; 16];
    session.copy_from_slice(&digest[..16]);
    session
}

fn required(name: &str) -> NativeExecutionResult<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(&format!("remote worker is missing {name}")))
}

fn parse_u64(name: &str) -> NativeExecutionResult<u64> {
    required(name)?
        .parse()
        .map_err(|_| invalid(&format!("{name} is malformed")))
}

fn invalid(message: &str) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.into())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
