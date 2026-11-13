use runmat_package::FrozenProjectHandoff;
use runmat_test::protocol::ProtocolLimits;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{NativeRunnerError, NativeRunnerResult};

pub const NATIVE_BOOTSTRAP_SCHEMA_VERSION: u16 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWorkerBootstrap {
    pub schema_version: u16,
    pub project: Option<FrozenProjectHandoff>,
    pub execution: crate::NativeWorkerExecutionConfig,
}

impl NativeWorkerBootstrap {
    pub fn new(
        project: Option<FrozenProjectHandoff>,
        execution: crate::NativeWorkerExecutionConfig,
    ) -> Self {
        Self {
            schema_version: NATIVE_BOOTSTRAP_SCHEMA_VERSION,
            project,
            execution,
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
    decode_bootstrap(&payload)
}

pub(crate) fn decode_bootstrap(payload: &[u8]) -> NativeRunnerResult<NativeWorkerBootstrap> {
    let bootstrap: NativeWorkerBootstrap = serde_json::from_slice(payload)
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_config::runtime::{
        AccelerateConfig, AccelerateProviderPreference, ForeignConfig, MexConfig,
        UnmanifestedMexPolicy,
    };

    #[tokio::test]
    async fn private_bootstrap_retains_resolved_execution_settings() {
        let bootstrap = NativeWorkerBootstrap::new(
            None,
            crate::NativeWorkerExecutionConfig {
                enable_jit: false,
                accelerate: AccelerateConfig {
                    enabled: true,
                    provider: AccelerateProviderPreference::Wgpu,
                    allow_inprocess_fallback: false,
                    ..AccelerateConfig::default()
                },
                foreign: ForeignConfig {
                    mex: MexConfig {
                        unmanifested: UnmanifestedMexPolicy::Deny,
                        ..MexConfig::default()
                    },
                    ..ForeignConfig::default()
                },
            },
        );
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        let limits = ProtocolLimits::default();
        let (written, read) = tokio::join!(
            write_bootstrap(&mut writer, &bootstrap, limits),
            read_bootstrap(&mut reader, limits),
        );
        written.unwrap();
        let restored = read.unwrap();
        assert_eq!(restored.schema_version, NATIVE_BOOTSTRAP_SCHEMA_VERSION);
        assert!(!restored.execution.enable_jit);
        assert!(restored.execution.accelerate.enabled);
        assert_eq!(
            restored.execution.accelerate.provider,
            AccelerateProviderPreference::Wgpu,
        );
        assert!(!restored.execution.accelerate.allow_inprocess_fallback);
        assert_eq!(
            restored.execution.foreign.mex.unmanifested,
            UnmanifestedMexPolicy::Deny,
        );
    }
}
