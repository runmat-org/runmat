use runmat_package::FrozenProjectHandoff;
use runmat_test::protocol::ProtocolLimits;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

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
    let length = u32::try_from(payload.len())
        .map_err(|_| NativeRunnerError::Protocol("bootstrap length exceeds u32".into()))?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_bootstrap(
    reader: &mut (impl AsyncRead + Unpin),
    limits: ProtocolLimits,
) -> NativeRunnerResult<NativeWorkerBootstrap> {
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header).await?;
    let length = super::framing::frame_length(header, limits)?;
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload).await?;
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
