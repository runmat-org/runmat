//! The live original node-agent observes a wave start, not the driver child.
//! Requests and signed replies use its existing allocation sandbox as local IPC.

use std::io::Read as _;
use std::path::Path;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_execution::security::{
    EndpointIdentitySigner, EndpointTrustPolicy, NodeClockEndV1, NodeClockOwnerV1,
    NodeClockStartRequestV1, NodeClockStartV1, MAX_CLOCK_OBSERVATIONS_PER_NODE,
    MAX_CLOCK_OBSERVATION_WIRE_BYTES,
};
use runmat_process_host::clock::observe_monotonic;

use crate::allocation::DriverClockScope;
use crate::enrollment::{decode_secret, NodeCredential};
use crate::{AgentError, AgentResult};
use runmat_execution_transport_native::control::{ClockNodeDemandV1, MAX_CLOCK_HEARTBEAT_ITEMS};

pub(super) fn pending_starts(state_directory: &Path) -> AgentResult<Vec<NodeClockStartV1>> {
    pending::<NodeClockStartV1>(&state_directory.join("clock-observations"))
}

pub(super) fn pending_ends(state_directory: &Path) -> AgentResult<Vec<NodeClockEndV1>> {
    pending::<NodeClockEndV1>(&state_directory.join("clock-ends"))
}

pub(super) fn pending_release_confirms(state_directory: &Path) -> AgentResult<Vec<[u8; 16]>> {
    let directory = state_directory.join("clock-ends");
    std::fs::create_dir_all(&directory)?;
    let mut files = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort();
    files
        .into_iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "confirmed")
        })
        .take(MAX_CLOCK_HEARTBEAT_ITEMS)
        .map(|path| {
            if !std::fs::symlink_metadata(&path)?.file_type().is_file() {
                return Err(reject("clock release confirmation is not a regular file"));
            }
            let end: NodeClockEndV1 = bounded_read(&path)?;
            end.validate_shape()
                .map_err(|error| reject(error.to_string()))?;
            let encoded = URL_SAFE_NO_PAD.encode(end.owner.observation_id);
            if path.file_name().and_then(|name| name.to_str())
                != Some(format!("{encoded}.confirmed").as_str())
            {
                return Err(reject("clock release confirmation differs from signed end"));
            }
            Ok(end.owner.observation_id)
        })
        .collect()
}

fn pending<T: serde::de::DeserializeOwned>(directory: &Path) -> AgentResult<Vec<T>> {
    std::fs::create_dir_all(directory)?;
    let mut files = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|value| value.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort();
    let end_directory = directory
        .file_name()
        .is_some_and(|name| name == "clock-ends");
    for path in files.iter().filter(|path| {
        path.file_name().is_some_and(|name| name == ".pending")
            || (end_directory
                && path
                    .extension()
                    .is_some_and(|extension| extension == "confirmed"))
    }) {
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file()
            || metadata.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES as u64
        {
            return Err(reject("clock observation has foreign or oversized staging"));
        }
    }
    files.retain(|path| {
        path.file_name().is_some_and(|name| name != ".pending")
            && !(end_directory
                && path
                    .extension()
                    .is_some_and(|extension| extension == "confirmed"))
    });
    files
        .into_iter()
        .take(MAX_CLOCK_HEARTBEAT_ITEMS)
        .map(|path| {
            if path.extension().is_none_or(|extension| extension != "json") {
                return Err(reject("clock observation has an unknown owned filename"));
            }
            if !std::fs::symlink_metadata(&path)?.file_type().is_file() {
                return Err(reject("clock observation contains a non-file"));
            }
            bounded_read(&path)
        })
        .collect()
}

pub(super) fn acknowledge(
    state_directory: &Path,
    start_ids: &[[u8; 16]],
    end_ids: &[[u8; 16]],
) -> AgentResult<()> {
    for id in start_ids {
        let path = state_directory
            .join("clock-observations")
            .join(format!("{}.json", URL_SAFE_NO_PAD.encode(id)));
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    // A same-directory write-through rename atomically changes a signed end
    // from retransmission authority to release-confirmation authority. A crash
    // cannot resurrect a sendable end after the Server removes its marker.
    let ends = state_directory.join("clock-ends");
    for id in end_ids {
        let encoded = URL_SAFE_NO_PAD.encode(id);
        let source = ends.join(format!("{encoded}.json"));
        let confirmed = ends.join(format!("{encoded}.confirmed"));
        if source.exists() {
            runmat_process_host::durable_file::publish_synced_file(&source, &confirmed)?;
        } else if !confirmed.exists() {
            return Err(reject("Server acknowledged an unknown node clock end"));
        }
    }
    Ok(())
}

pub(super) fn acknowledge_release_confirms(
    state_directory: &Path,
    ids: &[[u8; 16]],
) -> AgentResult<()> {
    for id in ids {
        let path = state_directory
            .join("clock-ends")
            .join(format!("{}.confirmed", URL_SAFE_NO_PAD.encode(id)));
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

pub(super) fn observe_ends(
    state_directory: &Path,
    credential: &NodeCredential,
    clock_domain_nonce: [u8; 16],
    demands: &[ClockNodeDemandV1],
) -> AgentResult<()> {
    if demands.len() > MAX_CLOCK_HEARTBEAT_ITEMS {
        return Err(reject("clock demand batch exceeds control budget"));
    }
    let directory = state_directory.join("clock-ends");
    std::fs::create_dir_all(&directory)?;
    let signer = EndpointIdentitySigner::from_secret(decode_secret(&credential.identity_secret)?)
        .map_err(|error| reject(error.to_string()))?;
    let policy = trust_policy(credential);
    for demand in demands {
        let start = &demand.start_receipt;
        policy
            .verify_node_clock_start(start, &credential.identity_fingerprint)
            .map_err(|error| reject(error.to_string()))?;
        if start.owner.org_id != credential.org_id
            || start.owner.cluster_id != credential.cluster_id
            || start.owner.node_id != credential.node_id
            || start.owner.clock_domain_nonce != clock_domain_nonce
            || demand.demand_challenge.iter().all(|byte| *byte == 0)
        {
            return Err(reject("clock demand differs from live original node owner"));
        }
        let destination = directory.join(format!(
            "{}.json",
            URL_SAFE_NO_PAD.encode(start.owner.observation_id)
        ));
        let confirmed = directory.join(format!(
            "{}.confirmed",
            URL_SAFE_NO_PAD.encode(start.owner.observation_id)
        ));
        if destination.exists() && confirmed.exists() {
            return Err(reject(
                "clock end has duplicate pending and confirmed owners",
            ));
        }
        if destination.exists() || confirmed.exists() {
            let prior: NodeClockEndV1 = bounded_read(if destination.exists() {
                &destination
            } else {
                &confirmed
            })?;
            if prior.owner != start.owner
                || prior.start_receipt_digest
                    != start
                        .receipt_digest()
                        .map_err(|error| reject(error.to_string()))?
                || prior.demand_challenge != demand.demand_challenge
            {
                return Err(reject(
                    "clock end demand changed after original observation",
                ));
            }
            continue;
        }
        let reading = observe_monotonic()?;
        if reading.ticks_per_second != start.owner.ticks_per_second
            || reading.ticks <= start.started_monotonic_ticks
        {
            return Err(reject("clock end is outside original live domain"));
        }
        let mut end = NodeClockEndV1 {
            schema_version: 1,
            owner: start.owner.clone(),
            start_receipt_digest: start
                .receipt_digest()
                .map_err(|error| reject(error.to_string()))?,
            demand_challenge: demand.demand_challenge,
            ended_monotonic_ticks: reading.ticks,
            signature: Vec::new(),
        };
        signer
            .sign_node_clock_end(&mut end)
            .map_err(|error| reject(error.to_string()))?;
        publish_json(&destination, &end)?;
    }
    Ok(())
}

pub(super) fn observe_starts(
    state_directory: &Path,
    credential: &NodeCredential,
    scopes: &[DriverClockScope],
) -> AgentResult<()> {
    let retained = state_directory.join("clock-observations");
    std::fs::create_dir_all(&retained)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&retained, std::fs::Permissions::from_mode(0o700))?;
    }
    let mut occupied = std::fs::read_dir(&retained)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|entry| entry.file_name() != ".pending")
        .count();
    if occupied > MAX_CLOCK_OBSERVATIONS_PER_NODE {
        return Err(reject(
            "node clock observation ownership exceeds checkpoint budget",
        ));
    }
    let signer = EndpointIdentitySigner::from_secret(decode_secret(&credential.identity_secret)?)
        .map_err(|error| reject(error.to_string()))?;
    let policy = trust_policy(credential);
    for scope in scopes {
        for file in std::fs::read_dir(&scope.request_directory)? {
            let file = file?;
            if file.file_name() == ".pending" {
                let metadata = std::fs::symlink_metadata(file.path())?;
                if !metadata.file_type().is_file()
                    || metadata.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES as u64
                {
                    return Err(reject("clock request has foreign or oversized staging"));
                }
                continue;
            }
            if !file.file_type()?.is_file() {
                return Err(reject("clock request is not an owned regular file"));
            }
            let mut bytes = Vec::with_capacity(MAX_CLOCK_OBSERVATION_WIRE_BYTES);
            std::fs::File::open(file.path())?
                .by_ref()
                .take((MAX_CLOCK_OBSERVATION_WIRE_BYTES + 1) as u64)
                .read_to_end(&mut bytes)?;
            if bytes.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES {
                return Err(reject("clock request exceeds its control budget"));
            }
            let request: NodeClockStartRequestV1 = serde_json::from_slice(&bytes)?;
            request
                .validate_shape()
                .map_err(|error| reject(error.to_string()))?;
            let encoded = URL_SAFE_NO_PAD.encode(request.observation_id);
            if file.file_name().to_str() != Some(format!("{encoded}.json").as_str())
                || request.run_id != scope.run_id
                || request.driver_allocation_lease_id != scope.allocation_lease_id
                || request.driver_allocation_fence != scope.allocation_fence
                || request.driver_process_launch_nonce != scope.process_launch_nonce
                || request.clock_domain_nonce != scope.clock_domain_nonce
            {
                return Err(reject("clock request differs from its live driver owner"));
            }
            let retained_path = retained.join(format!("{encoded}.json"));
            let start = if retained_path.exists() {
                let saved: NodeClockStartV1 = bounded_read(&retained_path)?;
                policy
                    .verify_node_clock_start(&saved, &credential.identity_fingerprint)
                    .map_err(|error| reject(error.to_string()))?;
                if saved.owner.org_id != credential.org_id
                    || saved.owner.cluster_id != credential.cluster_id
                    || saved.owner.node_id != credential.node_id
                    || saved.owner.identity_public_key != credential.identity_public_key
                    || saved.owner.identity_fingerprint != credential.identity_fingerprint
                    || saved.owner.run_id != scope.run_id
                    || saved.owner.driver_allocation_lease_id != scope.allocation_lease_id
                    || saved.owner.driver_allocation_fence != scope.allocation_fence
                    || saved.owner.observation_id != request.observation_id
                    || saved.owner.driver_process_launch_nonce != scope.process_launch_nonce
                    || saved.owner.clock_domain_nonce != scope.clock_domain_nonce
                {
                    return Err(reject("clock observation ID was reused across owners"));
                }
                saved
            } else {
                if occupied == MAX_CLOCK_OBSERVATIONS_PER_NODE {
                    // Leave the request in place: no work can begin without a
                    // signed reply and an acknowledged pre-dispatch checkpoint.
                    continue;
                }
                let reading = observe_monotonic()?;
                let mut start = NodeClockStartV1 {
                    schema_version: 1,
                    owner: NodeClockOwnerV1 {
                        org_id: credential.org_id.clone(),
                        cluster_id: credential.cluster_id.clone(),
                        node_id: credential.node_id.clone(),
                        run_id: scope.run_id.clone(),
                        driver_allocation_lease_id: scope.allocation_lease_id.clone(),
                        driver_allocation_fence: scope.allocation_fence,
                        driver_process_launch_nonce: scope.process_launch_nonce,
                        observation_id: request.observation_id,
                        clock_domain_nonce: scope.clock_domain_nonce,
                        ticks_per_second: reading.ticks_per_second,
                        identity_public_key: credential.identity_public_key.clone(),
                        identity_fingerprint: credential.identity_fingerprint.clone(),
                    },
                    started_monotonic_ticks: reading.ticks,
                    signature: Vec::new(),
                };
                signer
                    .sign_node_clock_start(&mut start)
                    .map_err(|error| reject(error.to_string()))?;
                publish_json(&retained_path, &start)?;
                occupied += 1;
                start
            };
            publish_json(
                &scope.receipt_directory.join(format!("{encoded}.json")),
                &start,
            )?;
            std::fs::remove_file(file.path())?;
        }
    }
    Ok(())
}

fn trust_policy(credential: &NodeCredential) -> EndpointTrustPolicy {
    EndpointTrustPolicy {
        permitted_tiers: Default::default(),
        trusted_identity_fingerprints: [credential.identity_fingerprint.clone()]
            .into_iter()
            .collect(),
        allowed_attestation_classes: Default::default(),
        require_pinned_identity: true,
        now_unix_millis: 0,
        maximum_clock_skew_millis: 0,
    }
}

fn bounded_read<T: serde::de::DeserializeOwned>(path: &Path) -> AgentResult<T> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES as u64 {
        return Err(reject(
            "clock observation is not an owned bounded regular file",
        ));
    }
    let mut bytes = Vec::with_capacity(MAX_CLOCK_OBSERVATION_WIRE_BYTES);
    std::fs::File::open(path)?
        .by_ref()
        .take((MAX_CLOCK_OBSERVATION_WIRE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES {
        return Err(reject("clock observation exceeds its control budget"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn publish_json<T: serde::Serialize>(destination: &Path, value: &T) -> AgentResult<()> {
    use std::io::Write as _;
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES {
        return Err(reject("clock observation exceeds its control budget"));
    }
    let staged = destination
        .parent()
        .ok_or_else(|| reject("clock observation lacks an owned staging parent"))?
        .join(".pending");
    if let Ok(metadata) = std::fs::symlink_metadata(&staged) {
        if !metadata.file_type().is_file()
            || metadata.len() > MAX_CLOCK_OBSERVATION_WIRE_BYTES as u64
        {
            return Err(reject("clock observation has foreign or oversized staging"));
        }
        // Staging is never admission or signing authority. An interrupted
        // publish may leave this one bounded slot; the retained final file is
        // checked independently before an idempotent retry.
        std::fs::remove_file(&staged)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    runmat_process_host::durable_file::publish_synced_file(&staged, destination)?;
    Ok(())
}

fn reject(message: impl Into<String>) -> AgentError {
    AgentError::AllocationRejected(message.into())
}

#[cfg(test)]
#[path = "clock_observer/tests.rs"]
mod tests;
