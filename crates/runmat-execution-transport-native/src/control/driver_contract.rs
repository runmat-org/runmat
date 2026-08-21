use async_trait::async_trait;

use super::worker_pool::DriverWorkerPool;
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
