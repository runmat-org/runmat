use serde::Deserialize;

#[cfg(test)]
pub const REMOTE_WORKER_PROTOCOL_V4: u16 = 4;
#[cfg(test)]
pub const REMOTE_WORKER_PROTOCOL_V5: u16 = 5;
#[cfg(test)]
pub const REMOTE_WORKER_PROTOCOL_V6: u16 = 6;
/// V7 requires the original typed cancellation reason on every Cancel command.
pub const REMOTE_WORKER_PROTOCOL_V7: u16 = 7;
pub const REMOTE_WORKER_PROTOCOL_VERSION: u16 = REMOTE_WORKER_PROTOCOL_V7;

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
        assert!(error.contains("unsupported remote worker protocol 4; expected 7"));
        assert!(error.contains("rebuild or upgrade"));
    }

    #[test]
    fn frozen_v5_is_rejected_before_command_deserialization() {
        let frozen = br#"{"schemaVersion":5,"command":{"kind":"drain"}}"#;
        let error = admit_remote_worker_bytes(frozen).unwrap_err();
        assert!(error.contains(&format!(
            "unsupported remote worker protocol {REMOTE_WORKER_PROTOCOL_V5}; expected 7"
        )));
    }

    #[test]
    fn v6_without_original_stop_cause_is_rejected() {
        assert_eq!(REMOTE_WORKER_PROTOCOL_V6, 6);
        let frozen = br#"{"schemaVersion":6,"command":{"kind":"drain"}}"#;
        assert!(admit_remote_worker_bytes(frozen)
            .unwrap_err()
            .contains("expected 7"));
    }

    #[test]
    fn v7_is_the_remote_wire_version() {
        assert_eq!(REMOTE_WORKER_PROTOCOL_V7, 7);
        admit_remote_worker_bytes(br#"{"schemaVersion":7,"command":{"kind":"drain"}}"#).unwrap();
    }
}
