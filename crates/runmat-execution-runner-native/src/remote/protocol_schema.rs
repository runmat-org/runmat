use serde::Deserialize;

#[cfg(test)]
pub const REMOTE_WORKER_PROTOCOL_V4: u16 = 4;
pub const REMOTE_WORKER_PROTOCOL_V5: u16 = 5;
pub const REMOTE_WORKER_PROTOCOL_VERSION: u16 = REMOTE_WORKER_PROTOCOL_V5;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteWorkerSchemaHeader {
    schema_version: u16,
}

pub fn admit_remote_worker_bytes(bytes: &[u8]) -> Result<(), String> {
    let header: RemoteWorkerSchemaHeader = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid remote worker envelope: {error}"))?;
    if header.schema_version != REMOTE_WORKER_PROTOCOL_VERSION {
        return Err(format!(
            "unsupported remote worker protocol {}; expected {}; rebuild or upgrade the remote worker",
            header.schema_version, REMOTE_WORKER_PROTOCOL_VERSION
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_v4_is_rejected_before_command_deserialization() {
        let frozen =
            br#"{"schemaVersion":4,"command":{"kind":"removed_output_list","values":[1]}}"#;
        let error = admit_remote_worker_bytes(frozen).unwrap_err();
        assert!(error.contains("unsupported remote worker protocol 4; expected 5"));
        assert!(error.contains("rebuild or upgrade"));
    }
}
