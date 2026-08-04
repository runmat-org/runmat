use async_trait::async_trait;
use base64::Engine as _;
use runmat_server_client::public_api::{self, types};

use super::ResourceRequest;
use crate::{TransportError, TransportResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverAuthority {
    pub server_url: String,
    pub run_id: String,
    pub org_id: String,
    pub project_id: String,
    pub allocation_lease_id: String,
    pub driver_lease_id: String,
    pub fencing_token: u64,
    pub credential: String,
}

impl DriverAuthority {
    pub fn validate(&self) -> TransportResult<()> {
        if self.server_url.trim().is_empty()
            || self.run_id.is_empty()
            || self.org_id.is_empty()
            || self.project_id.is_empty()
            || self.allocation_lease_id.is_empty()
            || self.driver_lease_id.is_empty()
            || self.fencing_token == 0
            || self.credential.is_empty()
            || self.credential.len() > 256
        {
            return Err(TransportError::StaleAuthority);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverArtifactDownload {
    pub artifact_id: String,
    pub kind: String,
    pub ciphertext_digest: String,
    pub ciphertext_size_bytes: u64,
    pub media_type: String,
    pub method: String,
    pub url: String,
    pub headers: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverBootstrap {
    pub project_revision: String,
    pub endpoint_fingerprint: String,
    pub run_key_envelope: Vec<u8>,
    pub artifacts: Vec<DriverArtifactDownload>,
    pub checkpoint: Option<DriverArtifactDownload>,
    pub cancellation_requested: bool,
    pub driver_resources: ResourceRequest,
    pub desired_worker_count: u32,
    pub worker_resources: ResourceRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverHeartbeat {
    pub run_state: String,
    pub cancellation_requested: bool,
    pub expires_at_millis: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverArtifactKind {
    Result,
    Checkpoint,
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointDisposition {
    ResumeSafe,
    Indeterminate,
}

pub struct StoreDriverArtifact<'a> {
    pub kind: DriverArtifactKind,
    pub ciphertext: &'a [u8],
    pub ciphertext_digest: String,
    pub media_type: String,
    pub retain_for_seconds: u64,
    pub checkpoint: Option<(u64, CheckpointDisposition)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredDriverArtifact {
    pub artifact_id: String,
    pub kind: String,
    pub ciphertext_size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverRunTarget {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Indeterminate,
}

pub struct DriverRunTransition {
    pub target: DriverRunTarget,
    pub reason_code: Option<String>,
    pub result_artifact_id: Option<String>,
    pub diagnostic_artifact_id: Option<String>,
}

use super::worker_pool::{self, DriverWorkerPool};

#[async_trait]
pub trait DriverControlPlane: Send + Sync {
    async fn bootstrap(&self, authority: &DriverAuthority) -> TransportResult<DriverBootstrap>;
    async fn heartbeat(
        &self,
        authority: &DriverAuthority,
        ttl_seconds: u64,
    ) -> TransportResult<DriverHeartbeat>;
    async fn record_usage(
        &self,
        authority: &DriverAuthority,
        source_sequence: u64,
    ) -> TransportResult<bool>;
    async fn download(&self, artifact: &DriverArtifactDownload) -> TransportResult<Vec<u8>>;
    async fn store_artifact(
        &self,
        authority: &DriverAuthority,
        artifact: StoreDriverArtifact<'_>,
    ) -> TransportResult<StoredDriverArtifact>;
    async fn transition(
        &self,
        authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> TransportResult<()>;
    async fn resize_workers(
        &self,
        authority: &DriverAuthority,
        expected_generation: u64,
        desired_workers: u32,
        resources: ResourceRequest,
    ) -> TransportResult<DriverWorkerPool> {
        let _ = (authority, expected_generation, desired_workers, resources);
        Err(TransportError::Unavailable(
            "remote worker allocation is not implemented".into(),
        ))
    }
    async fn list_workers(&self, authority: &DriverAuthority) -> TransportResult<DriverWorkerPool> {
        let _ = authority;
        Err(TransportError::Unavailable(
            "remote worker listing is not implemented".into(),
        ))
    }
    async fn authorize_worker(
        &self,
        authority: &DriverAuthority,
        allocation_lease_id: &str,
        endpoint_fingerprint: &str,
        run_key_envelope: &[u8],
    ) -> TransportResult<()> {
        let _ = (
            authority,
            allocation_lease_id,
            endpoint_fingerprint,
            run_key_envelope,
        );
        Err(TransportError::Unavailable(
            "remote worker authorization is not implemented".into(),
        ))
    }
}

#[derive(Clone)]
pub struct HttpDriverControlPlane {
    client: public_api::Client,
    http: reqwest::Client,
}

impl HttpDriverControlPlane {
    pub fn new(base_url: impl Into<String>) -> TransportResult<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(TransportError::Unavailable("Server URL is empty".into()));
        }
        Ok(Self {
            client: public_api::Client::new(&base_url),
            http: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl DriverControlPlane for HttpDriverControlPlane {
    async fn bootstrap(&self, authority: &DriverAuthority) -> TransportResult<DriverBootstrap> {
        authority.validate()?;
        let response = self
            .client
            .driver_bootstrap(
                &authority.driver_lease_id,
                to_i64(authority.fencing_token)?,
                &authority.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        if response.run_id != authority.run_id
            || response.org_id != authority.org_id
            || response.project_id != authority.project_id
            || response.allocation_lease_id != authority.allocation_lease_id
            || response.driver_lease_id != authority.driver_lease_id
            || to_u64(response.fencing_token)? != authority.fencing_token
        {
            return Err(TransportError::StaleAuthority);
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
            driver_resources: worker_pool::resource_from_api(response.driver_resources)?,
            desired_worker_count: u32::try_from(response.desired_worker_count)
                .map_err(|_| TransportError::Overflow)?,
            worker_resources: ResourceRequest {
                cpu_millicores: to_u64(response.worker_resources.cpu_millicores)?,
                memory_bytes: to_u64(response.worker_resources.memory_bytes)?,
                scratch_bytes: to_u64(response.worker_resources.scratch_bytes)?,
                accelerator_count: u32::try_from(response.worker_resources.accelerator_count)
                    .map_err(|_| TransportError::Overflow)?,
                accelerator_class: response.worker_resources.accelerator_class,
                accelerator_memory_bytes: to_u64(
                    response.worker_resources.accelerator_memory_bytes,
                )?,
                maximum_wall_millis: to_u64(response.worker_resources.maximum_wall_millis)?,
            },
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
                &authority.driver_lease_id,
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
                &authority.driver_lease_id,
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::DriverHeartbeatRequest {
                    ttl_seconds: to_i64(ttl_seconds)?,
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
                &authority.driver_lease_id,
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
                &authority.driver_lease_id,
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
                &authority.driver_lease_id,
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
                &authority.driver_lease_id,
                to_i64(authority.fencing_token)?,
                &authority.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        worker_pool::from_api(response)
    }

    async fn authorize_worker(
        &self,
        authority: &DriverAuthority,
        allocation_lease_id: &str,
        endpoint_fingerprint: &str,
        run_key_envelope: &[u8],
    ) -> TransportResult<()> {
        self.client
            .authorize_driver_worker(
                &authority.driver_lease_id,
                allocation_lease_id,
                to_i64(authority.fencing_token)?,
                &authority.credential,
                &types::AuthorizeDriverWorkerRequest {
                    endpoint_fingerprint: endpoint_fingerprint.to_string(),
                    run_key_envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .encode(run_key_envelope),
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
