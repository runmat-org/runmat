use async_trait::async_trait;
use base64::Engine as _;
use futures_util::StreamExt as _;
use runmat_server_client::public_api::{self, types};
use sha2::{Digest as _, Sha256};
use tokio::io::AsyncWriteExt as _;

use super::clock::{
    bounded as bounded_clock, decode_id as decode_clock_id, encode_id as encode_clock_id,
    MAX_CLOCK_HEARTBEAT_ITEMS,
};
use super::driver_contract::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverAuthority,
    DriverBootstrap, DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition,
    StoreDriverArtifact, StoredDriverArtifact, WorkerRetirementLookup,
};
use super::worker_pool::{self, DriverWorkerPool};
use super::ResourceRequest;
use super::{ClockDriverHeartbeatRequestV1, ClockDriverHeartbeatResponseV1};
use crate::{TransportError, TransportResult};

#[derive(Clone)]
pub struct HttpDriverControlPlane {
    base_url: String,
    client: public_api::Client,
    http: reqwest::Client,
}

#[cfg(test)]
mod clock_wire_bound_tests {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

    #[test]
    fn complete_release_manifest_fits_one_bounded_driver_heartbeat() {
        let entry = serde_json::json!({
            "observationId": URL_SAFE_NO_PAD.encode([u8::MAX; 16]),
            "startReceiptDigest": runmat_execution::Digest::sha256([u8::MAX; 32]),
            "endReceiptDigest": runmat_execution::Digest::sha256([u8::MAX; 64]),
        });
        let entries = vec![entry; runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE];
        let body = serde_json::json!({
            "ttlSeconds": 30,
            "clockStartReservations": vec![URL_SAFE_NO_PAD.encode([u8::MAX; 16]); 128],
            // A 4094-byte string occupies 4096 JSON bytes including quotes.
            "clockStartReceipts": vec!["x".repeat(4094); 128],
            "clockEndDemands": vec![serde_json::json!({
                "observationId": URL_SAFE_NO_PAD.encode([u8::MAX; 16]),
                "startReceiptDigest": runmat_execution::Digest::sha256([u8::MAX; 32]),
                "demandChallenge": URL_SAFE_NO_PAD.encode([u8::MAX; 16]),
            }); 128],
            "clockLookupIds": vec![URL_SAFE_NO_PAD.encode([u8::MAX; 16]); 128],
            "clockObservationReleaseManifest": {
                "checkpointSequence": u64::MAX,
                "checkpointArtifactId": "a".repeat(128),
                "checkpointCiphertextDigest": "b".repeat(128),
                "driverFence": u64::MAX,
                "entries": entries,
            },
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(bytes.len() < 2 * 1024 * 1024);
        // The per-entry bound describes the release manifest, not the whole
        // heartbeat: its independent receipt batch alone can occupy 512 KiB.
        let manifest_bytes = serde_json::to_vec(&body["clockObservationReleaseManifest"]).unwrap();
        assert!(
            manifest_bytes.len()
                <= runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE * 232 + 64 * 1024
        );
    }
}

impl HttpDriverControlPlane {
    pub fn new(base_url: impl Into<String>) -> TransportResult<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(TransportError::Unavailable("Server URL is empty".into()));
        }
        Ok(Self {
            client: public_api::Client::new(&base_url),
            base_url,
            http: reqwest::Client::new(),
        })
    }
}

/// A clone-cheap HTTP client for signed artifact URLs passed through an
/// authenticated driver-to-worker route. It carries no Server control-plane
/// credential and shares the exact streamed verifier with the driver.
#[derive(Clone, Default)]
pub struct SignedArtifactDownloader {
    http: reqwest::Client,
}

impl SignedArtifactDownloader {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn download_to(
        &self,
        artifact: &DriverArtifactDownload,
        writer: &mut (dyn tokio::io::AsyncWrite + Send + Unpin),
        maximum_ciphertext_bytes: u64,
    ) -> TransportResult<()> {
        download_signed_artifact_to(&self.http, artifact, writer, maximum_ciphertext_bytes).await
    }
}

/// Stream a server-signed ciphertext object into caller-owned private staging.
/// Partial output is never published by this helper.
async fn download_signed_artifact_to(
    http: &reqwest::Client,
    artifact: &DriverArtifactDownload,
    writer: &mut (dyn tokio::io::AsyncWrite + Send + Unpin),
    maximum_ciphertext_bytes: u64,
) -> TransportResult<()> {
    if artifact.method != "GET"
        || artifact.ciphertext_size_bytes == 0
        || artifact.ciphertext_size_bytes > maximum_ciphertext_bytes
    {
        return Err(TransportError::Integrity);
    }
    let mut request = http.get(&artifact.url);
    for (name, value) in &artifact.headers {
        request = request.header(name, value);
    }
    let response = request
        .send()
        .await
        .map_err(|error| TransportError::Unavailable(error.to_string()))?;
    if !response.status().is_success() {
        return Err(TransportError::Unavailable(format!(
            "artifact download returned {}",
            response.status()
        )));
    }
    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut received = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| TransportError::Unavailable(error.to_string()))?;
        received = received
            .checked_add(chunk.len() as u64)
            .ok_or(TransportError::Overflow)?;
        if received > artifact.ciphertext_size_bytes {
            return Err(TransportError::Integrity);
        }
        hasher.update(&chunk);
        writer
            .write_all(&chunk)
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
    }
    if received != artifact.ciphertext_size_bytes
        || format!("sha256:{}", hex::encode(hasher.finalize())) != artifact.ciphertext_digest
    {
        return Err(TransportError::Integrity);
    }
    writer
        .flush()
        .await
        .map_err(|error| TransportError::Unavailable(error.to_string()))?;
    Ok(())
}

#[async_trait]
impl DriverControlPlane for HttpDriverControlPlane {
    async fn bootstrap(&self, authority: &DriverAuthority) -> TransportResult<DriverBootstrap> {
        authority.validate()?;
        let response = self
            .client
            .driver_bootstrap(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        if response.run_id != authority.run_id
            || response.org_id != authority.org_id.as_str()
            || response.project_id != authority.project_id
            || response.allocation_lease_id != authority.allocation_lease_id
            || response.driver_lease_id != authority.driver_lease_id.as_str()
            || to_u64(response.fencing_token)? != authority.fencing_token
        {
            return Err(TransportError::StaleAuthority);
        }
        let driver_resources = worker_pool::resource_from_api(response.driver_resources)?;
        // Older Direct bootstraps omit this additive field, including runs
        // with accelerator requests. Direct retains its existing execution
        // route; absence confers no concrete device grant to Graph.
        let has_driver_accelerator_grant = response.driver_accelerator_devices.is_some();
        let driver_accelerator_devices = response
            .driver_accelerator_devices
            .unwrap_or_default()
            .into_iter()
            .map(runmat_server_client::execution::accelerator_device_from_api)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        if has_driver_accelerator_grant
            && !runmat_execution::resource::accelerator_devices_exactly_satisfy(
                &driver_accelerator_devices,
                &driver_resources.accelerators,
            )
        {
            return Err(TransportError::Integrity);
        }
        Ok(DriverBootstrap {
            project_revision: response.project_revision,
            endpoint_fingerprint: response.endpoint_fingerprint,
            run_key_envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(response.run_key_envelope)
                .map_err(|_| TransportError::Integrity)?,
            artifacts: response
                .artifacts
                .into_iter()
                .map(download_from_api)
                .collect::<TransportResult<_>>()?,
            checkpoint: response.checkpoint.map(download_from_api).transpose()?,
            cancellation_requested: response.cancellation_requested,
            driver_resources,
            driver_accelerator_devices,
            desired_worker_count: u32::try_from(response.desired_worker_count)
                .map_err(|_| TransportError::Overflow)?,
            worker_resources: worker_pool::resource_from_api(response.worker_resources)?,
        })
    }

    async fn record_usage(
        &self,
        authority: &DriverAuthority,
        source_sequence: u64,
    ) -> TransportResult<bool> {
        let response = self
            .client
            .record_driver_usage(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::DriverUsageRequest {
                    source_sequence: to_i64(source_sequence)?,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(response.accepted)
    }

    async fn heartbeat(
        &self,
        authority: &DriverAuthority,
        ttl_seconds: u64,
    ) -> TransportResult<DriverHeartbeat> {
        let response = self
            .client
            .driver_heartbeat(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::DriverHeartbeatRequest {
                    ttl_seconds: to_i64(ttl_seconds)?,
                    clock_start_receipts: None,
                    clock_start_reservations: None,
                    clock_end_demands: None,
                    clock_lookup_ids: None,
                    clock_observation_release_manifest: None,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(DriverHeartbeat {
            run_state: response.run_state,
            cancellation_requested: response.cancellation_requested,
            expires_at_millis: response.expires_at.timestamp_millis(),
        })
    }

    async fn clock_heartbeat(
        &self,
        authority: &DriverAuthority,
        ttl_seconds: u64,
        request: ClockDriverHeartbeatRequestV1,
    ) -> TransportResult<ClockDriverHeartbeatResponseV1> {
        // Two independent 128-item lookup/echo sources can yield 256 starts
        // and 256 ends. A single 4096-entry manifest is <=232 bytes/entry;
        // 128 KiB covers demand/ID arrays and JSON envelope punctuation.
        const MAX_CLOCK_RESPONSE_BYTES: usize = 4
            * MAX_CLOCK_HEARTBEAT_ITEMS
            * runmat_execution::security::MAX_CLOCK_OBSERVATION_WIRE_BYTES
            + runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE * 232
            + 128 * 1024;
        authority.validate()?;
        for count in [
            request.start_reservations.len(),
            request.start_receipts.len(),
            request.end_demands.len(),
            request.lookup_ids.len(),
        ] {
            bounded_clock(count)?;
        }
        if request.release_manifest.as_ref().is_some_and(|manifest| {
            manifest.entries.len() > runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE
        }) {
            return Err(TransportError::Overflow);
        }
        for start in &request.start_receipts {
            start
                .validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        }
        let manifest = request.release_manifest.as_ref().map(|value| {
            serde_json::json!({
                "checkpointSequence": value.checkpoint_sequence,
                "checkpointArtifactId": value.checkpoint_artifact_id,
                "checkpointCiphertextDigest": value.checkpoint_ciphertext_digest,
                "driverFence": value.driver_fence,
                "entries": value.entries.iter().map(|entry| serde_json::json!({
                    "observationId": encode_clock_id(&entry.observation_id),
                    "startReceiptDigest": entry.start_receipt_digest,
                    "endReceiptDigest": entry.end_receipt_digest,
                })).collect::<Vec<_>>(),
            })
        });
        let body = serde_json::json!({
            "ttlSeconds": ttl_seconds,
            "clockStartReservations": request.start_reservations.iter().map(encode_clock_id).collect::<Vec<_>>(),
            "clockStartReceipts": request.start_receipts,
            "clockEndDemands": request.end_demands.iter().map(|demand| serde_json::json!({
                "observationId": encode_clock_id(&demand.observation_id),
                "startReceiptDigest": demand.start_receipt_digest,
                "demandChallenge": encode_clock_id(&demand.demand_challenge),
            })).collect::<Vec<_>>(),
            "clockLookupIds": request.lookup_ids.iter().map(encode_clock_id).collect::<Vec<_>>(),
            "clockObservationReleaseManifest": manifest,
        });
        let response = self
            .http
            .post(format!(
                "{}/v1/execution/drivers/{}/heartbeat",
                self.base_url, authority.driver_lease_id
            ))
            .query(&[("fencingToken", authority.fencing_token)])
            .header("x-runmat-driver-credential", &authority.credential)
            .header("content-type", "application/json")
            .body(
                serde_json::to_vec(&body)
                    .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
            )
            .send()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        if !response.status().is_success() {
            return Err(TransportError::Unavailable(format!(
                "clock heartbeat returned {}",
                response.status()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| TransportError::Unavailable(error.to_string()))?;
            if bytes
                .len()
                .checked_add(chunk.len())
                .is_none_or(|total| total > MAX_CLOCK_RESPONSE_BYTES)
            {
                return Err(TransportError::FrameTooLarge);
            }
            bytes.extend_from_slice(&chunk);
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Wire {
            run_state: String,
            cancellation_requested: bool,
            expires_at: chrono::DateTime<chrono::Utc>,
            #[serde(default)]
            clock_starts: Vec<runmat_execution::security::NodeClockStartV1>,
            #[serde(default)]
            clock_ends: Vec<runmat_execution::security::NodeClockEndV1>,
            #[serde(default)]
            clock_pending_end_demands: Vec<WirePendingDemand>,
            #[serde(default)]
            clock_acknowledged_start_ids: Vec<String>,
            #[serde(default)]
            clock_acked_start_ids: Vec<String>,
            #[serde(default)]
            clock_reserved_start_ids: Vec<String>,
            #[serde(default)]
            clock_release_accepted: bool,
            #[serde(default)]
            clock_latest_accepted_release_manifest: Option<WireReleaseManifest>,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct WireReleaseManifest {
            checkpoint_sequence: u64,
            checkpoint_artifact_id: String,
            checkpoint_ciphertext_digest: String,
            driver_fence: u64,
            entries: Vec<WireReleaseEntry>,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct WireReleaseEntry {
            observation_id: String,
            start_receipt_digest: runmat_execution::Digest,
            end_receipt_digest: runmat_execution::Digest,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct WirePendingDemand {
            observation_id: String,
            start_receipt_digest: runmat_execution::Digest,
            demand_challenge: String,
        }
        let wire: Wire = serde_json::from_slice(&bytes)
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        let acked = if wire.clock_acked_start_ids.is_empty() {
            wire.clock_acknowledged_start_ids
        } else {
            wire.clock_acked_start_ids
        };
        for count in [
            wire.clock_starts.len(),
            wire.clock_ends.len(),
            wire.clock_pending_end_demands.len(),
            acked.len(),
            wire.clock_reserved_start_ids.len(),
        ] {
            if count > 2 * MAX_CLOCK_HEARTBEAT_ITEMS {
                return Err(TransportError::Overflow);
            }
        }
        if wire
            .clock_latest_accepted_release_manifest
            .as_ref()
            .is_some_and(|manifest| {
                manifest.entries.len() > runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE
            })
        {
            return Err(TransportError::Overflow);
        }
        for value in &wire.clock_starts {
            value
                .validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        }
        for value in &wire.clock_ends {
            value
                .validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        }
        Ok(ClockDriverHeartbeatResponseV1 {
            run: DriverHeartbeat {
                run_state: wire.run_state,
                cancellation_requested: wire.cancellation_requested,
                expires_at_millis: wire.expires_at.timestamp_millis(),
            },
            reserved_start_ids: wire
                .clock_reserved_start_ids
                .iter()
                .map(|value| decode_clock_id(value))
                .collect::<TransportResult<_>>()?,
            acked_start_ids: acked
                .iter()
                .map(|value| decode_clock_id(value))
                .collect::<TransportResult<_>>()?,
            starts: wire.clock_starts,
            ends: wire.clock_ends,
            pending_end_demands: wire
                .clock_pending_end_demands
                .into_iter()
                .map(|demand| {
                    Ok(super::ClockEndDemandV1 {
                        observation_id: decode_clock_id(&demand.observation_id)?,
                        start_receipt_digest: demand.start_receipt_digest,
                        demand_challenge: decode_clock_id(&demand.demand_challenge)?,
                    })
                })
                .collect::<TransportResult<_>>()?,
            release_accepted: wire.clock_release_accepted,
            latest_accepted_release_manifest: wire
                .clock_latest_accepted_release_manifest
                .map(|manifest| {
                    Ok(super::ClockObservationReleaseManifestV1 {
                        checkpoint_sequence: manifest.checkpoint_sequence,
                        checkpoint_artifact_id: manifest.checkpoint_artifact_id,
                        checkpoint_ciphertext_digest: manifest.checkpoint_ciphertext_digest,
                        driver_fence: manifest.driver_fence,
                        entries: manifest
                            .entries
                            .into_iter()
                            .map(|entry| {
                                Ok(super::ClockObservationReleaseEntryV1 {
                                    observation_id: decode_clock_id(&entry.observation_id)?,
                                    start_receipt_digest: entry.start_receipt_digest,
                                    end_receipt_digest: entry.end_receipt_digest,
                                })
                            })
                            .collect::<TransportResult<_>>()?,
                    })
                })
                .transpose()?,
        })
    }

    async fn download(&self, artifact: &DriverArtifactDownload) -> TransportResult<Vec<u8>> {
        if artifact.method != "GET" {
            return Err(TransportError::Integrity);
        }
        let mut request = self.http.get(&artifact.url);
        for (name, value) in &artifact.headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        if !response.status().is_success() {
            return Err(TransportError::Unavailable(format!(
                "artifact download returned {}",
                response.status()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?
            .to_vec();
        crate::transfer::verify_ciphertext(
            &crate::transfer::OpaqueObject {
                ciphertext_digest: artifact.ciphertext_digest.clone(),
                ciphertext_size_bytes: artifact.ciphertext_size_bytes,
            },
            &bytes,
        )?;
        Ok(bytes)
    }

    async fn execution_artifact(
        &self,
        authority: &DriverAuthority,
        artifact_id: &str,
    ) -> TransportResult<DriverArtifactDownload> {
        authority.validate()?;
        if artifact_id.is_empty()
            || artifact_id.len() > 128
            || !artifact_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(TransportError::Integrity);
        }
        let response = self
            .http
            .get(format!(
                "{}/v1/execution/drivers/{}/artifacts/{}",
                self.base_url, authority.driver_lease_id, artifact_id
            ))
            .query(&[("fencingToken", authority.fencing_token)])
            .header("x-runmat-driver-credential", &authority.credential)
            .send()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        if !response.status().is_success() {
            return Err(TransportError::Unavailable(format!(
                "execution artifact authorization returned {}",
                response.status()
            )));
        }
        let artifact = response
            .json::<types::DriverArtifactDownloadResponse>()
            .await
            .map_err(|_| TransportError::Integrity)?;
        download_from_api(artifact)
    }

    async fn download_to(
        &self,
        artifact: &DriverArtifactDownload,
        writer: &mut (dyn tokio::io::AsyncWrite + Send + Unpin),
        maximum_ciphertext_bytes: u64,
    ) -> TransportResult<()> {
        download_signed_artifact_to(&self.http, artifact, writer, maximum_ciphertext_bytes).await
    }

    async fn store_artifact(
        &self,
        authority: &DriverAuthority,
        artifact: StoreDriverArtifact<'_>,
    ) -> TransportResult<StoredDriverArtifact> {
        let (checkpoint_sequence, recovery_disposition) = match artifact.checkpoint {
            Some((sequence, disposition)) => (
                Some(to_i64(sequence)?),
                Some(match disposition {
                    CheckpointDisposition::ResumeSafe => {
                        types::CheckpointRecoveryDispositionRequest::ResumeSafe
                    }
                    CheckpointDisposition::Indeterminate => {
                        types::CheckpointRecoveryDispositionRequest::Indeterminate
                    }
                }),
            ),
            None => (None, None),
        };
        let response = self
            .client
            .store_artifact(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::StoreDriverArtifactRequest {
                    checkpoint_sequence,
                    ciphertext: base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .encode(artifact.ciphertext),
                    ciphertext_digest: artifact.ciphertext_digest,
                    kind: match artifact.kind {
                        DriverArtifactKind::Result => types::DriverArtifactKindRequest::Result,
                        DriverArtifactKind::Checkpoint => {
                            types::DriverArtifactKindRequest::Checkpoint
                        }
                        DriverArtifactKind::Diagnostic => {
                            types::DriverArtifactKindRequest::Diagnostic
                        }
                    },
                    media_type: artifact.media_type,
                    recovery_disposition,
                    retain_for_seconds: to_i64(artifact.retain_for_seconds)?,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(StoredDriverArtifact {
            artifact_id: response.artifact_id,
            kind: response.kind,
            ciphertext_size_bytes: to_u64(response.ciphertext_size_bytes)?,
        })
    }

    async fn transition(
        &self,
        authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> TransportResult<()> {
        self.client
            .driver_transition(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::DriverTransitionRequest {
                    diagnostic_artifact_id: transition.diagnostic_artifact_id,
                    reason_code: transition.reason_code,
                    result_artifact_id: transition.result_artifact_id,
                    target: match transition.target {
                        DriverRunTarget::Running => types::DriverTransitionTargetRequest::Running,
                        DriverRunTarget::Succeeded => {
                            types::DriverTransitionTargetRequest::Succeeded
                        }
                        DriverRunTarget::Failed => types::DriverTransitionTargetRequest::Failed,
                        DriverRunTarget::Cancelled => {
                            types::DriverTransitionTargetRequest::Cancelled
                        }
                        DriverRunTarget::Indeterminate => {
                            types::DriverTransitionTargetRequest::Indeterminate
                        }
                    },
                },
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn resize_workers(
        &self,
        authority: &DriverAuthority,
        expected_generation: u64,
        desired_workers: u32,
        resources: ResourceRequest,
    ) -> TransportResult<DriverWorkerPool> {
        let response = self
            .client
            .resize_driver_workers(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::ResizeDriverWorkersRequest {
                    expected_generation: to_i64(expected_generation)?,
                    desired_workers: i32::try_from(desired_workers)
                        .map_err(|_| TransportError::Overflow)?,
                    resources: worker_pool::resource_to_api(resources)?,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        worker_pool::from_api(response)
    }

    async fn list_workers(&self, authority: &DriverAuthority) -> TransportResult<DriverWorkerPool> {
        let response = self
            .client
            .list_driver_workers(
                authority.driver_lease_id.as_str(),
                to_i64(authority.fencing_token)?,
                &authority.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        worker_pool::from_api(response)
    }

    async fn worker_retirement(
        &self,
        authority: &DriverAuthority,
        original: &WorkerRetirementLookup,
    ) -> TransportResult<Option<runmat_execution::security::WorkerTreeRetirementEvidenceV1>> {
        authority.validate()?;
        if original.run_id != authority.run_id
            || original.allocation_lease_id.is_empty()
            || original.allocation_fencing_token == 0
            || original.driver_fencing_token == 0
        {
            return Err(TransportError::StaleAuthority);
        }
        let mut url = reqwest::Url::parse(&self.base_url)
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        url.path_segments_mut()
            .map_err(|_| {
                TransportError::Unavailable("driver server URL cannot hold path segments".into())
            })?
            .extend([
                "v1",
                "execution",
                "drivers",
                authority.driver_lease_id.as_str(),
                "workers",
                &original.allocation_lease_id,
                "retirement",
            ]);
        let response = self
            .http
            .get(url)
            .header("X-RunMat-Driver-Credential", &authority.credential)
            .query(&[
                ("fencingToken", authority.fencing_token.to_string()),
                (
                    "originalAllocationFencingToken",
                    original.allocation_fencing_token.to_string(),
                ),
                (
                    "originalDriverFencingToken",
                    original.driver_fencing_token.to_string(),
                ),
            ])
            .send()
            .await
            .map_err(|error| TransportError::Unavailable(error.to_string()))?;
        if !response.status().is_success() {
            return Err(TransportError::Unavailable(format!(
                "worker retirement retrieval status {}",
                response.status()
            )));
        }
        // The response wraps one bounded certificate as {"evidence": ...}.
        // 64 bytes cover that fixed JSON envelope without excluding a
        // certificate at the exact shared wire limit.
        const MAX_RETIREMENT_RESPONSE_BYTES: usize =
            runmat_execution::security::MAX_RETIREMENT_WIRE_BYTES + 64;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RETIREMENT_RESPONSE_BYTES as u64)
        {
            return Err(TransportError::MalformedFrame(
                "worker retirement response exceeds bound".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(16 * 1024);
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| TransportError::Unavailable(error.to_string()))?;
            if chunk.len() > MAX_RETIREMENT_RESPONSE_BYTES - bytes.len() {
                return Err(TransportError::MalformedFrame(
                    "worker retirement response exceeds bound".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct WireEvidence {
            schema_version: u16,
            org_id: String,
            cluster_id: String,
            node_id: String,
            run_id: String,
            allocation_lease_id: String,
            allocation_fencing_token: u64,
            driver_fencing_token: u64,
            process_id: u32,
            process_launch_nonce: String,
            retired_at_unix_millis: u64,
            clock_domain_nonce: String,
            retired_monotonic_ticks: u64,
            ticks_per_second: u64,
            encrypted_attempt_receipts: Vec<String>,
            identity_public_key: String,
            identity_fingerprint: String,
            signature: String,
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Response {
            evidence: Option<WireEvidence>,
        }
        let value: Response = serde_json::from_slice(&bytes)
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
        let evidence = value
            .evidence
            .map(|wire| {
                let decode = |value: &str| {
                    base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .decode(value)
                        .map_err(|error| TransportError::MalformedFrame(error.to_string()))
                };
                let process_launch_nonce: [u8; 16] = decode(&wire.process_launch_nonce)?
                    .try_into()
                    .map_err(|_| {
                        TransportError::MalformedFrame("invalid retirement launch nonce".into())
                    })?;
                Ok::<_, TransportError>(
                    runmat_execution::security::WorkerTreeRetirementEvidenceV1 {
                        schema_version: wire.schema_version,
                        org_id: wire.org_id,
                        cluster_id: wire.cluster_id,
                        node_id: wire.node_id,
                        run_id: wire.run_id,
                        allocation_lease_id: wire.allocation_lease_id,
                        allocation_fencing_token: wire.allocation_fencing_token,
                        driver_fencing_token: wire.driver_fencing_token,
                        process_id: wire.process_id,
                        process_launch_nonce,
                        retired_at_unix_millis: wire.retired_at_unix_millis,
                        clock_domain_nonce: decode(&wire.clock_domain_nonce)?.try_into().map_err(
                            |_| {
                                TransportError::MalformedFrame(
                                    "invalid retirement clock nonce".into(),
                                )
                            },
                        )?,
                        retired_monotonic_ticks: wire.retired_monotonic_ticks,
                        ticks_per_second: wire.ticks_per_second,
                        encrypted_attempt_receipts: wire.encrypted_attempt_receipts,
                        identity_public_key: decode(&wire.identity_public_key)?,
                        identity_fingerprint: wire.identity_fingerprint,
                        signature: decode(&wire.signature)?,
                    },
                )
            })
            .transpose()?;
        if let Some(evidence) = &evidence {
            evidence
                .validate_shape()
                .map_err(|error| TransportError::MalformedFrame(error.to_string()))?;
            if evidence.run_id != original.run_id
                || evidence.allocation_lease_id != original.allocation_lease_id
                || evidence.allocation_fencing_token != original.allocation_fencing_token
                || evidence.driver_fencing_token != original.driver_fencing_token
            {
                return Err(TransportError::StaleAuthority);
            }
        }
        Ok(evidence)
    }

    async fn authorize_worker(
        &self,
        authority: &DriverAuthority,
        allocation_lease_id: &str,
        endpoint_fingerprint: &str,
        run_key_envelope: &[u8],
        ready_v1: bool,
    ) -> TransportResult<()> {
        self.client
            .authorize_driver_worker(
                authority.driver_lease_id.as_str(),
                allocation_lease_id,
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::AuthorizeDriverWorkerRequest {
                    endpoint_fingerprint: endpoint_fingerprint.to_string(),
                    run_key_envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .encode(run_key_envelope),
                    // An older Server rejects unknown request fields. The
                    // NodeAgent forwards an explicit support bit from a new
                    // Server's driver bootstrap; otherwise omit the field.
                    ready_v1: ready_v1.then_some(true),
                },
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }
}

fn download_from_api(
    value: types::DriverArtifactDownloadResponse,
) -> TransportResult<DriverArtifactDownload> {
    Ok(DriverArtifactDownload {
        artifact_id: value.artifact_id,
        kind: value.kind,
        ciphertext_digest: value.ciphertext_digest,
        ciphertext_size_bytes: to_u64(value.ciphertext_size_bytes)?,
        media_type: value.media_type,
        method: value.method,
        url: value.url,
        headers: value.headers.into_iter().collect(),
    })
}

fn to_i64(value: u64) -> TransportResult<i64> {
    i64::try_from(value).map_err(|_| TransportError::Overflow)
}

fn to_u64(value: i64) -> TransportResult<u64> {
    u64::try_from(value).map_err(|_| TransportError::Overflow)
}

fn map_error<E: std::fmt::Debug>(error: public_api::Error<E>) -> TransportError {
    if error
        .status()
        .is_some_and(|status| matches!(status.as_u16(), 401 | 403 | 404 | 409))
    {
        TransportError::StaleAuthority
    } else {
        TransportError::Unavailable(error.to_string())
    }
}

#[cfg(test)]
mod streamed_download_tests {
    use std::time::Duration;

    use sha2::{Digest as _, Sha256};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    use super::{DriverArtifactDownload, SignedArtifactDownloader};
    use crate::TransportError;

    fn descriptor(url: String, size: u64, digest: &str) -> DriverArtifactDownload {
        DriverArtifactDownload {
            artifact_id: "signed-object".into(),
            kind: "program".into(),
            ciphertext_digest: digest.into(),
            ciphertext_size_bytes: size,
            media_type: "application/octet-stream".into(),
            method: "GET".into(),
            url,
            headers: Default::default(),
        }
    }

    async fn serve_once(listener: TcpListener, body: &'static [u8]) {
        let (mut stream, _) = listener.accept().await.expect("loopback request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.ends_with(b"\r\n\r\n") {
            let count = stream.read(&mut buffer).await.expect("request read");
            assert!(count > 0 && request.len() + count < 16 * 1024);
            request.extend_from_slice(&buffer[..count]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        assert!(request.starts_with(b"GET "));
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .expect("response header");
        stream.write_all(body).await.expect("response body");
        stream.shutdown().await.expect("response close");
    }

    #[tokio::test]
    async fn signed_stream_checks_exact_size_and_digest_before_caller_publication() {
        let cases: [(&str, &'static [u8], u64, bool); 4] = [
            ("valid", b"sealed", 6, true),
            ("truncated", b"seal", 6, false),
            ("overrun", b"sealed!", 6, false),
            ("wrong_digest", b"sealed", 6, false),
        ];
        for (name, body, declared_size, succeeds) in cases {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("loopback listener");
            let url = format!("http://{}/signed", listener.local_addr().expect("address"));
            let mut server = tokio::spawn(serve_once(listener, body));
            let digest = if name == "wrong_digest" {
                format!("sha256:{}", hex::encode(Sha256::digest(b"different")))
            } else {
                format!("sha256:{}", hex::encode(Sha256::digest(b"sealed")))
            };
            let artifact = descriptor(url, declared_size, &digest);
            let (mut writer, mut reader) = tokio::io::duplex(128);
            let result = match tokio::time::timeout(
                Duration::from_secs(5),
                SignedArtifactDownloader::new().download_to(&artifact, &mut writer, 32),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    server.abort();
                    let _ = server.await;
                    panic!("{name}: signed download timed out");
                }
            };
            drop(writer);
            match tokio::time::timeout(Duration::from_secs(5), &mut server).await {
                Ok(result) => result.expect("loopback server task"),
                Err(_) => {
                    server.abort();
                    let _ = server.await;
                    panic!("{name}: loopback server did not retire");
                }
            }
            let mut staged = Vec::new();
            tokio::time::timeout(Duration::from_secs(5), reader.read_to_end(&mut staged))
                .await
                .expect("staged read timeout")
                .expect("staged bytes");
            if succeeds {
                assert_eq!(result, Ok(()), "{name}");
                assert_eq!(staged, body, "{name}");
            } else {
                assert_eq!(result, Err(TransportError::Integrity), "{name}");
                if name == "overrun" {
                    assert!(
                        staged.len() <= declared_size as usize,
                        "overrun must not write bytes past the declared bound"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn signed_stream_rejects_bound_before_any_http_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let url = format!("http://{}/signed", listener.local_addr().expect("address"));
        let artifact = descriptor(
            url,
            7,
            &format!("sha256:{}", hex::encode(Sha256::digest(b"sealed"))),
        );
        let (mut writer, _reader) = tokio::io::duplex(128);
        assert_eq!(
            SignedArtifactDownloader::new()
                .download_to(&artifact, &mut writer, 6)
                .await,
            Err(TransportError::Integrity)
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err()
        );
    }
}
