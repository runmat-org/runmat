#![cfg(unix)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use runmat_execution_transport_native::control::{
    DriverBootstrapCredential, EnrolledNode, EnrollmentRequest, NodeAllocation, NodeControlPlane,
    NodeHeartbeat, NodeStatus, ResourceRequest, RotatedCredential,
};
use runmat_execution_transport_native::TransportResult;
use runmat_node_agent::enrollment::{CredentialStore, NodeCredential};
use runmat_node_agent::service::NodeAgentService;
use runmat_node_agent::AgentConfig;

#[derive(Default)]
struct State {
    lease_state: String,
    node_state: String,
    releases: usize,
    drain_completions: usize,
}

struct Control {
    state: Mutex<State>,
}

#[async_trait]
impl NodeControlPlane for Control {
    async fn enroll(&self, _: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        unreachable!()
    }

    async fn heartbeat(&self, _: NodeHeartbeat) -> TransportResult<NodeStatus> {
        let state = self.state.lock().unwrap();
        Ok(NodeStatus {
            state: state.node_state.clone(),
            credential_epoch: 1,
            lease_epoch: 1,
        })
    }

    async fn rotate_credential(&self, _: &NodeHeartbeat) -> TransportResult<RotatedCredential> {
        unreachable!()
    }

    async fn allocations(&self, _: &NodeHeartbeat) -> TransportResult<Vec<NodeAllocation>> {
        let state = self.state.lock().unwrap();
        if state.lease_state == "released" {
            Ok(Vec::new())
        } else {
            Ok(vec![NodeAllocation {
                id: "lease-1".into(),
                run_id: "run-1".into(),
                project_id: "project-1".into(),
                queue: "default".into(),
                resources: ResourceRequest {
                    cpu_millicores: 1,
                    memory_bytes: 8 * 1024 * 1024 * 1024,
                    scratch_bytes: 1024,
                    accelerator_count: 0,
                    accelerator_class: None,
                    accelerator_memory_bytes: 0,
                    maximum_wall_millis: 10_000,
                },
                state: state.lease_state.clone(),
                fencing_token: 1,
                expires_at_millis: 4_000_000_000_000,
            }])
        }
    }

    async fn accept(&self, _: &NodeHeartbeat, _: &NodeAllocation) -> TransportResult<()> {
        self.state.lock().unwrap().lease_state = "active".into();
        Ok(())
    }

    async fn publish_endpoint_identity(
        &self,
        _: &NodeHeartbeat,
        allocation: &NodeAllocation,
        evidence: runmat_execution::security::EndpointIdentityEvidence,
    ) -> TransportResult<()> {
        assert_eq!(evidence.allocation_lease_id, allocation.id);
        assert_eq!(evidence.run_identity, allocation.run_id);
        Ok(())
    }

    async fn driver_bootstrap(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<DriverBootstrapCredential> {
        Ok(DriverBootstrapCredential {
            run_id: allocation.run_id.clone(),
            org_id: heartbeat.org_id.clone(),
            project_id: allocation.project_id.clone(),
            allocation_lease_id: allocation.id.clone(),
            driver_lease_id: "driver-lease-1".into(),
            fencing_token: 1,
            credential: "driver-credential".into(),
            expires_at_millis: allocation.expires_at_millis,
        })
    }

    async fn release(&self, _: &NodeHeartbeat, _: &NodeAllocation) -> TransportResult<()> {
        let mut state = self.state.lock().unwrap();
        state.lease_state = "released".into();
        state.releases += 1;
        Ok(())
    }

    async fn complete_drain(&self, _: &NodeHeartbeat) -> TransportResult<()> {
        self.state.lock().unwrap().drain_completions += 1;
        Ok(())
    }
}

#[tokio::test]
async fn service_reaps_fixed_mode_process_releases_lease_and_completes_drain() {
    let directory = tempfile::tempdir().unwrap();
    let identity_secret = [19; 32];
    let signer =
        runmat_execution::security::EndpointIdentitySigner::from_secret(identity_secret).unwrap();
    CredentialStore::new(directory.path())
        .store(&NodeCredential {
            node_id: "node-1".into(),
            cluster_id: "cluster-1".into(),
            org_id: "org-1".into(),
            identity_secret: base64::Engine::encode(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                identity_secret,
            ),
            identity_public_key: signer.public_key().to_vec(),
            identity_fingerprint: signer.fingerprint(),
            credential: "c".repeat(43),
            credential_epoch: 1,
            lease_epoch: 1,
        })
        .unwrap();
    let control = Arc::new(Control {
        state: Mutex::new(State {
            lease_state: "offered".into(),
            node_state: "active".into(),
            ..State::default()
        }),
    });
    let mut service = NodeAgentService::load(
        AgentConfig {
            state_directory: directory.path().to_path_buf(),
            server_url: "http://127.0.0.1:1".into(),
            runmat_executable: "/usr/bin/true".into(),
            heartbeat_interval: Duration::from_millis(10),
            heartbeat_ttl: Duration::from_secs(10),
            drain_timeout: Duration::from_millis(10),
            maximum_allocations: 1,
            trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        },
        control.clone(),
    )
    .unwrap();

    service.reconcile_once().await.unwrap();
    service.reconcile_once().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    service.reconcile_once().await.unwrap();
    {
        let state = control.state.lock().unwrap();
        assert_eq!(state.releases, 1);
    }
    control.state.lock().unwrap().node_state = "draining".into();
    service.reconcile_once().await.unwrap();
    assert_eq!(control.state.lock().unwrap().drain_completions, 1);
}
