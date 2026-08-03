use runmat_package::FrozenProjectHandoff;
use runmat_test::protocol::ProtocolLimits;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{NativeRunnerError, NativeRunnerResult};

pub const NATIVE_BOOTSTRAP_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWorkerBootstrap {
    pub schema_version: u16,
    pub project: Option<FrozenProjectHandoff>,
}

impl NativeWorkerBootstrap {
    pub fn new(project: Option<FrozenProjectHandoff>) -> Self {
        Self {
            schema_version: NATIVE_BOOTSTRAP_SCHEMA_VERSION,
            project,
        }
    }
}

pub async fn write_bootstrap(
    writer: &mut (impl AsyncWrite + Unpin),
    bootstrap: &NativeWorkerBootstrap,
    limits: ProtocolLimits,
) -> NativeRunnerResult<()> {
    let payload = serde_json::to_vec(bootstrap)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    if payload.len() > limits.max_message_bytes as usize {
        return Err(NativeRunnerError::Protocol(
            "native worker bootstrap exceeds the negotiated message bound".into(),
        ));
    }
    runmat_process_host::ipc::write_payload(writer, &payload, super::framing::host_limits(limits))
        .await
        .map_err(map_host_error)?;
    Ok(())
}

pub async fn read_bootstrap(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<NativeWorkerBootstrap> {
    let payload =
        runmat_process_host::ipc::read_payload(reader, super::framing::host_limits(limits))
            .await
            .map_err(map_host_error)?;
    let bootstrap: NativeWorkerBootstrap = serde_json::from_slice(&payload)
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    if bootstrap.schema_version != NATIVE_BOOTSTRAP_SCHEMA_VERSION {
        return Err(NativeRunnerError::Protocol(format!(
            "unsupported native bootstrap schema {}; supported schema is {}",
            bootstrap.schema_version, NATIVE_BOOTSTRAP_SCHEMA_VERSION
        )));
    }
    if let Some(project) = &bootstrap.project {
        project
            .validate()
            .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    }
    Ok(bootstrap)
}

fn map_host_error(error: runmat_process_host::ProcessHostError) -> NativeRunnerError {
    match error {
        runmat_process_host::ProcessHostError::Io(error) => NativeRunnerError::Io(error),
        error => NativeRunnerError::Protocol(error.to_string()),
    }
}
