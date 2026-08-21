use std::collections::{HashMap, HashSet};

use rand::RngCore as _;
use runmat_execution::identity::{PoolId, WorkerId};
use runmat_execution::security::{EndpointTrustPolicy, ExecutionTrustTier};
use runmat_execution_artifact::encryption::{
    encode_run_key_envelope, ExecutionHpkeSuite, ExecutionRecipientKey,
    PortableExecutionEncryption, RunKeyMaterial,
};
use runmat_execution_runner::WorkerSpec;
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, DriverWorkerAllocation, DriverWorkerPool,
    ResourceRequest as AllocationResources,
};
use runmat_execution_transport_native::frame::FrameLimits;

use super::pool_resources::inventory;
use super::worker_env::session_id;
use super::{RelayRemoteWorkerChannel, RemotePoolDriver};
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) async fn reconcile_workers(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    run_key: &RunKeyMaterial,
    pool_id: PoolId,
    pool: &std::sync::Arc<RemotePoolDriver>,
    worker_pool: &DriverWorkerPool,
    registered: &mut HashMap<String, WorkerId>,
) -> NativeExecutionResult<()> {
    let live_allocations = worker_pool
        .workers
        .iter()
        .filter(|worker| matches!(worker.state.as_str(), "offered" | "active"))
        .map(|worker| worker.allocation_lease_id.as_str())
        .collect::<HashSet<_>>();
    let lost = registered
        .iter()
        .filter(|(allocation, _)| !live_allocations.contains(allocation.as_str()))
        .map(|(allocation, worker)| (allocation.clone(), *worker))
        .collect::<Vec<_>>();
    for (allocation, worker) in lost {
        pool.remove_worker(worker, true).await?;
        registered.remove(&allocation);
    }

    for worker in worker_pool
        .workers
        .iter()
        .filter(|worker| live_allocations.contains(worker.allocation_lease_id.as_str()))
    {
        if registered.contains_key(&worker.allocation_lease_id) {
            continue;
        }
        let Some(evidence) = worker.endpoint_identity.as_ref() else {
            continue;
        };
        verify_evidence(authority, worker)?;
        if !worker.run_key_envelope_authorized {
            authorize_worker(control, authority, run_key, worker).await?;
        }
        let worker_id = WorkerId::derive(&[
            authority.run_id.as_bytes(),
            worker.allocation_lease_id.as_bytes(),
        ]);
        let channel = RelayRemoteWorkerChannel::connect(
            &worker_relay_url(authority, &worker.allocation_lease_id)?,
            &[
                (
                    "X-RunMat-Driver-Credential".into(),
                    authority.credential.clone(),
                ),
                (
                    "Sec-WebSocket-Protocol".into(),
                    "runmat-worker-relay-v1".into(),
                ),
            ],
            super::RemoteWorkerChannelConfig {
                run_identity: authority.run_id.clone(),
                node_identity: evidence.node_id.clone(),
                worker: WorkerSpec {
                    id: worker_id,
                    pool_id,
                    resources: inventory(&worker.resources)?,
                },
                driver_fence: authority.fencing_token,
                session_id: session_id(&authority.run_id, &worker.allocation_lease_id),
                run_key: run_key.clone(),
                limits: FrameLimits::default(),
            },
        )
        .await?;
        pool.add_worker(channel).await?;
        registered.insert(worker.allocation_lease_id.clone(), worker_id);
    }
    Ok(())
}

async fn authorize_worker(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    run_key: &RunKeyMaterial,
    worker: &DriverWorkerAllocation,
) -> NativeExecutionResult<()> {
    let evidence = worker
        .endpoint_identity
        .as_ref()
        .ok_or_else(|| protocol("worker endpoint identity is missing"))?;
    let recipient = ExecutionRecipientKey {
        suite: ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
        fingerprint: evidence.recipient.fingerprint.clone(),
        public_key: evidence.recipient.public_key.clone(),
        valid_after_unix_millis: evidence.recipient.valid_after_unix_millis,
        valid_before_unix_millis: evidence.recipient.valid_before_unix_millis,
        custodian_uri: None,
    };
    let mut entropy = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut entropy);
    let envelope = PortableExecutionEncryption
        .seal_run_key_with_entropy(entropy, &recipient, run_key, authority.run_id.clone(), 1)
        .map_err(protocol)?;
    control
        .authorize_worker(
            authority,
            &worker.allocation_lease_id,
            &recipient.fingerprint,
            &encode_run_key_envelope(&envelope).map_err(protocol)?,
        )
        .await
        .map_err(protocol)
}

pub(super) fn validate_pool_intent(
    pool: &DriverWorkerPool,
    desired_workers: u32,
    resources: &AllocationResources,
) -> NativeExecutionResult<()> {
    if pool.generation == 0
        || pool.desired_workers != desired_workers
        || &pool.resources != resources
    {
        return Err(protocol("Server returned a substituted worker pool intent"));
    }
    Ok(())
}

fn verify_evidence(
    authority: &DriverAuthority,
    worker: &DriverWorkerAllocation,
) -> NativeExecutionResult<()> {
    let evidence = worker
        .endpoint_identity
        .as_ref()
        .ok_or_else(|| protocol("worker endpoint identity is missing"))?;
    if evidence.org_id != authority.org_id
        || evidence.run_identity != authority.run_id
        || evidence.allocation_lease_id != worker.allocation_lease_id
        || evidence.fencing_token != worker.fencing_token
    {
        return Err(protocol(
            "worker endpoint evidence is outside driver authority",
        ));
    }
    let mut policy = EndpointTrustPolicy {
        permitted_tiers: [evidence.trust_tier].into_iter().collect(),
        trusted_identity_fingerprints: [evidence.identity_fingerprint.clone()]
            .into_iter()
            .collect(),
        allowed_attestation_classes: Default::default(),
        require_pinned_identity: true,
        now_unix_millis: u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(protocol)?
                .as_millis(),
        )
        .map_err(|_| protocol("system clock exceeds endpoint time range"))?,
        maximum_clock_skew_millis: 30_000,
    };
    if evidence.trust_tier == ExecutionTrustTier::AttestedConfidential {
        if let Some(class) = evidence.attestation_class.clone() {
            policy.allowed_attestation_classes.insert(class);
        }
    }
    policy.verify(evidence).map_err(protocol)
}

fn worker_relay_url(
    authority: &DriverAuthority,
    allocation_id: &str,
) -> NativeExecutionResult<String> {
    let base = authority.server_url.trim_end_matches('/');
    let base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        return Err(protocol("driver Server URL must use HTTP or HTTPS"));
    };
    Ok(format!(
        "{base}/v1/execution/drivers/{}/workers/{allocation_id}/relay?fencingToken={}",
        authority.driver_lease_id, authority.fencing_token
    ))
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
