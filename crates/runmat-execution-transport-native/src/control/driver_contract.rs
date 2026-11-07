use async_trait::async_trait;

use super::worker_pool::DriverWorkerPool;
use super::ResourceRequest;
use crate::{TransportError, TransportResult};
use runmat_server_client::ids::{ExecutionDriverLeaseId, OrgId};

/// Server-issued authority for one remote driver lease. The organization and
/// driver-lease identifiers are typed, so an authority that holds a malformed
/// identifier cannot be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverAuthority {
    pub server_url: String,
    pub run_id: String,
    pub org_id: OrgId,
    pub project_id: String,
    pub allocation_lease_id: String,
    pub driver_lease_id: ExecutionDriverLeaseId,
    pub fencing_token: u64,
    pub credential: String,
}

impl DriverAuthority {
    /// Parses a Server-issued organization identifier (`org_` + 32 lowercase hex).
    pub fn parse_org_id(value: &str) -> TransportResult<OrgId> {
        OrgId::parse(value).map_err(|_| TransportError::MalformedAuthority("org id"))
    }

    /// Parses a Server-issued driver-lease identifier (`drv_` + 32 lowercase hex).
    pub fn parse_driver_lease_id(value: &str) -> TransportResult<ExecutionDriverLeaseId> {
        ExecutionDriverLeaseId::parse(value)
            .map_err(|_| TransportError::MalformedAuthority("driver lease id"))
    }

    pub fn validate(&self) -> TransportResult<()> {
        if self.server_url.trim().is_empty()
            || self.run_id.is_empty()
            || self.project_id.is_empty()
            || self.allocation_lease_id.is_empty()
            || self.fencing_token == 0
            || self.credential.is_empty()
            || self.credential.len() > 256
        {
            return Err(TransportError::StaleAuthority);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Exact active driver-allocation devices from authenticated bootstrap.
    pub driver_accelerator_devices: Vec<runmat_execution::resource::AcceleratorDevice>,
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

/// Exact original grant, never inferred from absence in a worker listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRetirementLookup {
    pub run_id: String,
    pub allocation_lease_id: String,
    pub allocation_fencing_token: u64,
    pub driver_fencing_token: u64,
}

#[async_trait]
pub trait DriverControlPlane: Send + Sync {
    async fn bootstrap(&self, authority: &DriverAuthority) -> TransportResult<DriverBootstrap>;
    async fn heartbeat(
        &self,
        authority: &DriverAuthority,
        ttl_seconds: u64,
    ) -> TransportResult<DriverHeartbeat>;
    async fn clock_heartbeat(
        &self,
        authority: &DriverAuthority,
        ttl_seconds: u64,
        request: super::ClockDriverHeartbeatRequestV1,
    ) -> TransportResult<super::ClockDriverHeartbeatResponseV1> {
        let _ = (authority, ttl_seconds, request);
        Err(TransportError::Unavailable(
            "durable node clock observation is unsupported".into(),
        ))
    }
    async fn record_usage(
        &self,
        authority: &DriverAuthority,
        source_sequence: u64,
    ) -> TransportResult<bool>;
    async fn download(&self, artifact: &DriverArtifactDownload) -> TransportResult<Vec<u8>>;
    async fn execution_artifact(
        &self,
        authority: &DriverAuthority,
        artifact_id: &str,
    ) -> TransportResult<DriverArtifactDownload> {
        let _ = (authority, artifact_id);
        Err(TransportError::Unavailable(
            "run-scoped execution artifact retrieval is not implemented".into(),
        ))
    }
    /// Stream a bounded Graph Program chunk into caller-owned private staging.
    /// The destination is not admitted until the exact ciphertext size/digest
    /// have been verified. Direct artifacts continue to use `download`.
    async fn download_to(
        &self,
        artifact: &DriverArtifactDownload,
        writer: &mut (dyn tokio::io::AsyncWrite + Send + Unpin),
        maximum_ciphertext_bytes: u64,
    ) -> TransportResult<()> {
        let _ = (artifact, writer, maximum_ciphertext_bytes);
        Err(TransportError::Unavailable(
            "streamed Graph Program download is unavailable".into(),
        ))
    }
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
    async fn worker_retirement(
        &self,
        authority: &DriverAuthority,
        original: &WorkerRetirementLookup,
    ) -> TransportResult<Option<runmat_execution::security::WorkerTreeRetirementEvidenceV1>> {
        let _ = (authority, original);
        Err(TransportError::Unavailable(
            "authenticated worker retirement retrieval is unsupported".into(),
        ))
    }
    async fn authorize_worker(
        &self,
        authority: &DriverAuthority,
        allocation_lease_id: &str,
        endpoint_fingerprint: &str,
        run_key_envelope: &[u8],
        ready_v1: bool,
    ) -> TransportResult<()> {
        let _ = (
            authority,
            allocation_lease_id,
            endpoint_fingerprint,
            run_key_envelope,
            ready_v1,
        );
        Err(TransportError::Unavailable(
            "remote worker authorization is not implemented".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_identifiers_parse_in_the_server_issued_form() {
        let org = DriverAuthority::parse_org_id("org_00000000000000000000000000000001").unwrap();
        assert_eq!(org.as_str(), "org_00000000000000000000000000000001");
        let lease =
            DriverAuthority::parse_driver_lease_id("drv_0000000000000000000000000000000a").unwrap();
        assert_eq!(lease.as_str(), "drv_0000000000000000000000000000000a");
    }

    #[test]
    fn malformed_authority_identifiers_name_the_field() {
        for value in [
            "org-1",
            "00000000000000000000000000000001",
            "drv_00000000000000000000000000000001",
        ] {
            let error = DriverAuthority::parse_org_id(value).unwrap_err();
            assert_eq!(error, TransportError::MalformedAuthority("org id"));
            assert_eq!(error.to_string(), "driver authority has a malformed org id");
        }
        for value in [
            "driver-1",
            "drv_0000000000000000000000000000000A",
            "org_00000000000000000000000000000001",
        ] {
            let error = DriverAuthority::parse_driver_lease_id(value).unwrap_err();
            assert_eq!(error, TransportError::MalformedAuthority("driver lease id"));
            assert_eq!(
                error.to_string(),
                "driver authority has a malformed driver lease id"
            );
        }
    }
}
