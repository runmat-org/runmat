use std::io::{Read as _, Write as _};
use std::path::PathBuf;
use std::time::Duration;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_execution::security::{
    EndpointTrustPolicy, NodeClockStartRequestV1, NodeClockStartV1,
    MAX_CLOCK_OBSERVATION_WIRE_BYTES,
};

use runmat_execution_transport_native::control::DriverAuthority;
use runmat_execution_transport_native::control::{
    ClockDriverHeartbeatRequestV1, DriverControlPlane,
};

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) struct RemoteDriverConfig {
    pub authority: DriverAuthority,
    pub endpoint_identity_file: PathBuf,
    pub ready_v1: bool,
    pub clock_parent: Option<DriverClockParent>,
}

#[derive(Clone)]
pub(super) struct DriverClockParent {
    pub(super) allocation_fence: u64,
    pub(super) process_launch_nonce: [u8; 16],
    pub(super) clock_domain_nonce: [u8; 16],
    pub(super) node_fingerprint: String,
    pub(super) request_directory: PathBuf,
    pub(super) receipt_directory: PathBuf,
}

impl DriverClockParent {
    pub(super) async fn begin(
        &self,
        authority: &DriverAuthority,
        control: &dyn DriverControlPlane,
    ) -> NativeExecutionResult<NodeClockStartV1> {
        let observation_id: [u8; 16] = loop {
            let value: [u8; 16] = rand::random();
            if value.iter().any(|byte| *byte != 0) {
                break value;
            }
        };
        let request = NodeClockStartRequestV1 {
            schema_version: 1,
            run_id: authority.run_id.clone(),
            driver_allocation_lease_id: authority.allocation_lease_id.clone(),
            driver_allocation_fence: self.allocation_fence,
            driver_process_launch_nonce: self.process_launch_nonce,
            observation_id,
            clock_domain_nonce: self.clock_domain_nonce,
        };
        request.validate_shape().map_err(protocol)?;
        let reserved = control
            .clock_heartbeat(
                authority,
                30,
                ClockDriverHeartbeatRequestV1 {
                    start_reservations: vec![observation_id],
                    ..Default::default()
                },
            )
            .await
            .map_err(protocol)?;
        if !reserved.reserved_start_ids.contains(&observation_id)
            || reserved.run.cancellation_requested
            || reserved.run.run_state != "running"
        {
            return Err(protocol(
                "Server did not reserve original node clock observation",
            ));
        }
        let name = format!("{}.json", URL_SAFE_NO_PAD.encode(observation_id));
        let destination = self.request_directory.join(&name);
        let staged = self.request_directory.join(".pending");
        let bytes = serde_json::to_vec(&request).map_err(protocol)?;
        if bytes.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES {
            return Err(protocol("clock start request exceeds control budget"));
        }
        if let Ok(metadata) = std::fs::symlink_metadata(&staged) {
            if !metadata.file_type().is_file()
                || metadata.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES as u64
            {
                return Err(protocol(
                    "clock start request has foreign or oversized staging",
                ));
            }
            std::fs::remove_file(&staged).map_err(protocol)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged)
            .map_err(protocol)?;
        file.write_all(&bytes).map_err(protocol)?;
        file.sync_all().map_err(protocol)?;
        runmat_process_host::durable_file::publish_synced_file(&staged, &destination)
            .map_err(protocol)?;
        let response = self.receipt_directory.join(name);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(protocol(
                    "original node clock owner did not acknowledge wave start",
                ));
            }
            match std::fs::File::open(&response) {
                Ok(mut file) => {
                    let mut bytes = Vec::with_capacity(MAX_CLOCK_OBSERVATION_WIRE_BYTES);
                    std::io::Read::by_ref(&mut file)
                        .take((MAX_CLOCK_OBSERVATION_WIRE_BYTES + 1) as u64)
                        .read_to_end(&mut bytes)
                        .map_err(protocol)?;
                    if bytes.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES {
                        return Err(protocol("node clock start exceeds control budget"));
                    }
                    let start: NodeClockStartV1 =
                        serde_json::from_slice(&bytes).map_err(protocol)?;
                    let policy = EndpointTrustPolicy {
                        permitted_tiers: Default::default(),
                        trusted_identity_fingerprints: [self.node_fingerprint.clone()]
                            .into_iter()
                            .collect(),
                        allowed_attestation_classes: Default::default(),
                        require_pinned_identity: true,
                        now_unix_millis: 0,
                        maximum_clock_skew_millis: 0,
                    };
                    policy
                        .verify_node_clock_start(&start, &self.node_fingerprint)
                        .map_err(protocol)?;
                    if start.owner.run_id != authority.run_id
                        || start.owner.driver_allocation_lease_id != authority.allocation_lease_id
                        || start.owner.driver_allocation_fence != self.allocation_fence
                        || start.owner.driver_process_launch_nonce != self.process_launch_nonce
                        || start.owner.clock_domain_nonce != self.clock_domain_nonce
                        || start.owner.observation_id != observation_id
                    {
                        return Err(protocol(
                            "node clock start differs from original driver parent",
                        ));
                    }
                    let accepted = control
                        .clock_heartbeat(
                            authority,
                            30,
                            ClockDriverHeartbeatRequestV1 {
                                start_receipts: vec![start.clone()],
                                ..Default::default()
                            },
                        )
                        .await
                        .map_err(protocol)?;
                    if !accepted.starts.iter().any(|stored| stored == &start)
                        || accepted.run.cancellation_requested
                        || accepted.run.run_state != "running"
                    {
                        return Err(protocol(
                            "Server did not durably register original node clock start",
                        ));
                    }
                    drop(file);
                    // The Server now owns the immutable signed receipt. The
                    // sandbox reply is only a bounded local handoff slot, not
                    // a lifetime journal for arbitrarily many waves.
                    std::fs::remove_file(&response).map_err(protocol)?;
                    return Ok(start);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(error) => return Err(protocol(error)),
            }
        }
    }
}

impl RemoteDriverConfig {
    pub fn from_env() -> NativeExecutionResult<Self> {
        let authority = authority_from(|name| std::env::var(name).ok())?;
        let endpoint_identity_file =
            PathBuf::from(required("RUNMAT_EXECUTION_ENDPOINT_IDENTITY_FILE")?);
        if !endpoint_identity_file.is_absolute() || !endpoint_identity_file.is_file() {
            return Err(invalid(
                "endpoint identity file must be an existing absolute file",
            ));
        }
        Ok(Self {
            authority,
            endpoint_identity_file,
            ready_v1: std::env::var("RUNMAT_EXECUTION_SERVER_READY_V1")
                .ok()
                .as_deref()
                == Some("1"),
            clock_parent: Some(DriverClockParent {
                allocation_fence: required("RUNMAT_EXECUTION_DRIVER_ALLOCATION_FENCE")?
                    .parse()
                    .map_err(|_| invalid("driver allocation fence is malformed"))?,
                process_launch_nonce: nonce("RUNMAT_EXECUTION_DRIVER_PROCESS_NONCE")?,
                clock_domain_nonce: nonce("RUNMAT_EXECUTION_CLOCK_DOMAIN_NONCE")?,
                node_fingerprint: required("RUNMAT_EXECUTION_PARENT_NODE_FINGERPRINT")?,
                request_directory: absolute_directory("RUNMAT_EXECUTION_CLOCK_REQUEST_DIRECTORY")?,
                receipt_directory: absolute_directory("RUNMAT_EXECUTION_CLOCK_RECEIPT_DIRECTORY")?,
            }),
        })
    }
}

/// Reads and validates the Server-issued driver authority. A malformed
/// organization or driver-lease identifier fails here, before any Server call.
fn authority_from(var: impl Fn(&str) -> Option<String>) -> NativeExecutionResult<DriverAuthority> {
    let required = |name: &str| {
        var(name)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid(&format!("remote driver is missing {name}")))
    };
    let authority = DriverAuthority {
        server_url: required("RUNMAT_EXECUTION_SERVER_URL")?,
        run_id: required("RUNMAT_EXECUTION_RUN_ID")?,
        org_id: DriverAuthority::parse_org_id(&required("RUNMAT_EXECUTION_ORG_ID")?)
            .map_err(|error| invalid(&error.to_string()))?,
        project_id: required("RUNMAT_EXECUTION_PROJECT_ID")?,
        allocation_lease_id: required("RUNMAT_EXECUTION_ALLOCATION_ID")?,
        driver_lease_id: DriverAuthority::parse_driver_lease_id(&required(
            "RUNMAT_EXECUTION_DRIVER_LEASE_ID",
        )?)
        .map_err(|error| invalid(&error.to_string()))?,
        fencing_token: required("RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN")?
            .parse()
            .map_err(|_| invalid("driver fencing token is malformed"))?,
        credential: required("RUNMAT_EXECUTION_DRIVER_CREDENTIAL")?,
    };
    authority
        .validate()
        .map_err(|error| invalid(&error.to_string()))?;
    Ok(authority)
}

fn nonce(name: &str) -> NativeExecutionResult<[u8; 16]> {
    let bytes = URL_SAFE_NO_PAD.decode(required(name)?).map_err(protocol)?;
    let value: [u8; 16] = bytes
        .try_into()
        .map_err(|_| invalid("clock nonce length is invalid"))?;
    if value.iter().all(|byte| *byte == 0) {
        return Err(invalid("clock nonce is zero"));
    }
    Ok(value)
}

fn absolute_directory(name: &str) -> NativeExecutionResult<PathBuf> {
    let path = PathBuf::from(required(name)?);
    if !path.is_absolute() || !path.is_dir() {
        return Err(invalid("clock observation directory is invalid"));
    }
    Ok(path)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

fn required(name: &str) -> NativeExecutionResult<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(&format!("remote driver is missing {name}")))
}

fn invalid(message: &str) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn variables(org_id: &str, driver_lease_id: &str) -> BTreeMap<&'static str, String> {
        [
            ("RUNMAT_EXECUTION_SERVER_URL", "https://control.invalid"),
            ("RUNMAT_EXECUTION_RUN_ID", "run"),
            ("RUNMAT_EXECUTION_ORG_ID", org_id),
            ("RUNMAT_EXECUTION_PROJECT_ID", "project"),
            ("RUNMAT_EXECUTION_ALLOCATION_ID", "allocation"),
            ("RUNMAT_EXECUTION_DRIVER_LEASE_ID", driver_lease_id),
            ("RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN", "3"),
            ("RUNMAT_EXECUTION_DRIVER_CREDENTIAL", "credential"),
        ]
        .into_iter()
        .map(|(name, value)| (name, value.to_string()))
        .collect()
    }

    fn read(org_id: &str, driver_lease_id: &str) -> NativeExecutionResult<DriverAuthority> {
        let variables = variables(org_id, driver_lease_id);
        authority_from(|name| variables.get(name).cloned())
    }

    fn configuration_message(result: NativeExecutionResult<DriverAuthority>) -> String {
        match result {
            Err(NativeExecutionError::Configuration(message)) => message,
            Err(other) => panic!("expected a configuration error, got {other}"),
            Ok(_) => panic!("expected a configuration error, got an authority"),
        }
    }

    #[test]
    fn server_issued_authority_is_typed_at_validation() {
        let org_id = crate::remote::test_ids::TEST_ORG_ID;
        let lease = crate::remote::test_ids::test_driver_lease_id(3);
        let authority = read(org_id, lease.as_str()).unwrap();
        assert_eq!(authority.org_id, crate::remote::test_ids::test_org_id());
        assert_eq!(authority.driver_lease_id, lease);
        assert_eq!(authority.fencing_token, 3);
    }

    #[test]
    fn malformed_org_id_is_rejected_at_validation() {
        let lease = crate::remote::test_ids::test_driver_lease_id(3);
        for org_id in ["org-1", "00000000000000000000000000000001", "org_ABC"] {
            assert_eq!(
                configuration_message(read(org_id, lease.as_str())),
                "driver authority has a malformed org id",
            );
        }
    }

    #[test]
    fn malformed_driver_lease_id_is_rejected_at_validation() {
        let org_id = crate::remote::test_ids::TEST_ORG_ID;
        for lease in ["driver-3", "00000000000000000000000000000003", org_id] {
            assert_eq!(
                configuration_message(read(org_id, lease)),
                "driver authority has a malformed driver lease id",
            );
        }
    }
}
