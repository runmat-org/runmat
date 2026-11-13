use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use chrono::Utc;
use runmat_execution::security::{EndpointIdentitySigner, WorkerTreeRetirementEvidenceV1};
use runmat_execution_transport_native::control::{
    AllocationRole, NodeControlPlane, NodeHeartbeat, ReconnectBackoff,
};
use runmat_process_host::ProcessSpawnEvidence;

use super::clock_observer::{
    acknowledge as acknowledge_clocks, acknowledge_release_confirms, observe_ends, observe_starts,
    pending_ends, pending_release_confirms, pending_starts,
};
use super::retirement::{self, PendingRetirementRecord};
use crate::allocation::{
    prepare as prepare_sandbox, prepare_endpoint_identity, retired_encrypted_receipts,
    validate_active, validate_allocation_set, validate_offer, AllocationLaunchContext,
    AllocationProcesses, DrainState, WorkerRetirementScope,
};
use crate::enrollment::{decode_secret, CredentialStore, NodeCredential, OwnedCredentialStore};
use crate::{inventory, AgentConfig, AgentError, AgentResult};

pub struct NodeAgentService {
    config: AgentConfig,
    control: Arc<dyn NodeControlPlane>,
    credential: NodeCredential,
    processes: AllocationProcesses,
    drain: DrainState,
    pending_release: BTreeMap<String, PendingRetirementRecord>,
    launched_worker_leases: BTreeMap<String, LaunchLedgerRecord>,
    /// Live process-generation observation domain; never restored from disk.
    clock_domain_nonce: [u8; 16],
    // Declared last: process/state fields drop before the credential writer lease.
    store: OwnedCredentialStore,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LaunchLedgerRecord {
    fencing_token: u64,
    certified_acknowledged: bool,
}

impl NodeAgentService {
    pub fn load(config: AgentConfig, control: Arc<dyn NodeControlPlane>) -> AgentResult<Self> {
        config.validate()?;
        let store = CredentialStore::new(&config.state_directory).claim()?;
        let credential = store.load()?;
        if crate::config::ServerEndpoint::parse(&config.server_url)? != credential.server_endpoint {
            return Err(AgentError::Configuration(
                "node credential belongs to a different Server endpoint".into(),
            ));
        }
        let pending_release = retirement::load(
            &config.state_directory,
            &credential,
            config.maximum_allocations,
        )?;
        let launched_worker_leases = load_launch_ledger(&config.state_directory)?;
        if launched_worker_leases.len() > config.maximum_allocations
            || pending_release.len() > config.maximum_allocations
            || launched_worker_leases
                .keys()
                .chain(pending_release.keys())
                .collect::<BTreeSet<_>>()
                .len()
                > config.maximum_allocations
        {
            return Err(AgentError::AllocationRejected(
                "retained worker ownership exceeds node capacity".into(),
            ));
        }
        Ok(Self {
            config,
            control,
            store,
            credential,
            processes: AllocationProcesses::default(),
            drain: DrainState::Accepting,
            pending_release,
            launched_worker_leases,
            clock_domain_nonce: loop {
                let nonce: [u8; 16] = rand::random();
                if nonce.iter().any(|value| *value != 0) {
                    break nonce;
                }
            },
        })
    }

    pub async fn run(
        mut self,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> AgentResult<()> {
        let mut heartbeat_tick = tokio::time::interval(self.config.heartbeat_interval);
        heartbeat_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut resource_tick = tokio::time::interval(std::time::Duration::from_millis(500));
        resource_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut reconnect = ReconnectBackoff::new(
            self.config.heartbeat_interval,
            self.config
                .heartbeat_ttl
                .saturating_sub(self.config.heartbeat_interval),
        )?;
        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        self.drain.begin();
                        break;
                    }
                }
                _ = heartbeat_tick.tick() => {
                    match self.reconcile_once().await {
                        Ok(()) => reconnect.reset(),
                        Err(AgentError::Transport(_)) => {
                            tokio::time::sleep(reconnect.next_delay()).await;
                        }
                        Err(error) => return Err(error),
                    }
                }
                _ = resource_tick.tick() => {
                    observe_starts(&self.config.state_directory, &self.credential, &self.processes.driver_clock_scopes())?;
                    let state_directory = &self.config.state_directory;
                    let maximum_allocations = self.config.maximum_allocations;
                    let pending_release = &mut self.pending_release;
                    let mut prepare = |allocation_id: &str, scope: Option<&WorkerRetirementScope>| {
                        retirement::prepare(
                            state_directory,
                            maximum_allocations,
                            pending_release,
                            allocation_id,
                            scope,
                        )
                    };
                    let retired = self.processes.enforce_local_limits(
                        Utc::now().timestamp_millis(),
                        &mut prepare,
                    ).await?;
                    self.record_retired(retired)?;
                }
            }
        }
        self.shutdown().await
    }

    pub async fn reconcile_once(&mut self) -> AgentResult<()> {
        let inventory = inventory::collect(self.config.trust_tier)?;
        let heartbeat = self.heartbeat(inventory.clone())?;
        let clock_response = self
            .control
            .clock_heartbeat(
                heartbeat.clone(),
                pending_starts(&self.config.state_directory)?,
                pending_ends(&self.config.state_directory)?,
                pending_release_confirms(&self.config.state_directory)?,
            )
            .await?;
        let status = clock_response.node;
        if status.credential_epoch != self.credential.credential_epoch {
            return Err(runmat_execution_transport_native::TransportError::StaleAuthority.into());
        }
        acknowledge_clocks(
            &self.config.state_directory,
            &clock_response.acked_start_ids,
            &clock_response.acked_end_ids,
        )?;
        acknowledge_release_confirms(
            &self.config.state_directory,
            &clock_response.acked_release_confirm_ids,
        )?;
        if self.credential.lease_epoch != status.lease_epoch {
            self.credential.lease_epoch = status.lease_epoch;
            self.store.store(&self.credential)?;
        }
        if status.state == "draining" {
            self.drain.begin();
        } else if status.state != "active" {
            let state_directory = &self.config.state_directory;
            let maximum_allocations = self.config.maximum_allocations;
            let pending_release = &mut self.pending_release;
            let mut prepare = |allocation_id: &str, scope: Option<&WorkerRetirementScope>| {
                retirement::prepare(
                    state_directory,
                    maximum_allocations,
                    pending_release,
                    allocation_id,
                    scope,
                )
            };
            let retired = self.processes.terminate_all(&mut prepare).await?;
            self.record_retired(retired)?;
            return Err(AgentError::AllocationRejected(format!(
                "node state {} cannot run allocations",
                status.state
            )));
        }
        observe_ends(
            &self.config.state_directory,
            &self.credential,
            self.clock_domain_nonce,
            &clock_response.demands,
        )?;
        let allocations = self.control.allocations(&heartbeat).await?;
        // A certified release ACK does not make a stale desired snapshot safe
        // to relaunch. Retain the single-launch tombstone until a subsequent
        // authoritative listing actually excludes the old lease.
        let before = self.launched_worker_leases.len();
        self.launched_worker_leases.retain(|id, record| {
            !record.certified_acknowledged
                || self.processes.contains(id)
                || self.pending_release.contains_key(id)
                || allocations
                    .iter()
                    .any(|lease| lease.id == *id && lease.fencing_token == record.fencing_token)
        });
        if self.launched_worker_leases.len() != before {
            self.persist_launch_ledger()?;
        }
        let now_millis = Utc::now().timestamp_millis();
        let state_directory = &self.config.state_directory;
        let maximum_allocations = self.config.maximum_allocations;
        let pending_release = &mut self.pending_release;
        let mut prepare = |allocation_id: &str, scope: Option<&WorkerRetirementScope>| {
            retirement::prepare(
                state_directory,
                maximum_allocations,
                pending_release,
                allocation_id,
                scope,
            )
        };
        let stale = self
            .processes
            .fence_stale(&allocations, now_millis, &mut prepare)
            .await?;
        self.record_retired(stale)?;
        validate_allocation_set(&allocations, &inventory, now_millis)?;
        let mut released_this_pass = BTreeSet::new();
        let state_directory = &self.config.state_directory;
        let maximum_allocations = self.config.maximum_allocations;
        let pending_release = &mut self.pending_release;
        let mut prepare = |allocation_id: &str, scope: Option<&WorkerRetirementScope>| {
            retirement::prepare(
                state_directory,
                maximum_allocations,
                pending_release,
                allocation_id,
                scope,
            )
        };
        let finished = self.processes.reap_finished(&mut prepare).await?;
        self.record_retired(finished)?;
        let pending = self.pending_release.clone();
        for (allocation_id, pending) in pending {
            let PendingRetirementRecord::Certified { evidence } = pending else {
                // A restart between durable intent and certified retirement
                // cannot recreate the old monotonic clock domain. Retain the
                // original ownership and capacity fail-closed.
                continue;
            };
            let allocation = allocations
                .iter()
                .find(|allocation| allocation.id == allocation_id);
            match (allocation, evidence.as_ref()) {
                (Some(allocation), Some(evidence)) => {
                    self.control
                        .release_retired_worker(&heartbeat, allocation, evidence)
                        .await?;
                }
                (None, Some(evidence)) => {
                    // The signed evidence carries the original lease and
                    // fencing authority, so a certified retry does not depend
                    // on the lease remaining in the active listing.
                    self.control
                        .release_retired_worker_evidence(&heartbeat, evidence)
                        .await?;
                }
                (Some(allocation), None) => {
                    self.control.release(&heartbeat, allocation).await?;
                }
                (None, None) => {
                    // Ordinary driver release has no certified worker record.
                    // Authoritative absence is its terminal release state.
                }
            }
            released_this_pass.insert(allocation_id.clone());
            if let Some(evidence) = &evidence {
                cleanup_retired_attempt_journal(&self.config.state_directory, &allocation_id)?;
                if let Some(record) = self.launched_worker_leases.get_mut(&allocation_id) {
                    if record.fencing_token == evidence.allocation_fencing_token {
                        record.certified_acknowledged = true;
                    }
                }
                self.persist_launch_ledger()?;
            }
            self.pending_release.remove(&allocation_id);
            self.persist_pending_release()?;
            // Keep the launch tombstone through this and any stale listing;
            // only an observed post-ACK absence above removes it.
        }
        if self.drain != DrainState::Accepting {
            return self
                .finish_drain(&heartbeat, clock_response.open_observer_count)
                .await;
        }
        for allocation in &allocations {
            if self.occupied_allocation_slots() >= self.config.maximum_allocations {
                break;
            }
            if allocation.state != "active"
                || self.processes.contains(&allocation.id)
                || released_this_pass.contains(&allocation.id)
                || self.pending_release.contains_key(&allocation.id)
            {
                continue;
            }
            validate_active(allocation, &inventory, now_millis)?;
            let sandbox = prepare_sandbox(&self.config.state_directory, allocation, &inventory)?;
            let spawn_evidence = ProcessSpawnEvidence::default();
            let launch = match allocation.role {
                AllocationRole::Driver => {
                    let bootstrap =
                        match self.control.driver_bootstrap(&heartbeat, allocation).await {
                            Ok(bootstrap) => bootstrap,
                            Err(runmat_execution_transport_native::TransportError::NotReady) => {
                                continue
                            }
                            Err(error) => return Err(error.into()),
                        };
                    validate_driver_bootstrap(
                        &self.credential,
                        allocation,
                        &bootstrap,
                        now_millis,
                    )?;
                    self.processes
                        .launch_driver(
                            AllocationLaunchContext {
                                runmat_executable: &self.config.runmat_executable,
                                allocation,
                                sandbox: &sandbox,
                                server_url: &self.config.server_url,
                                clock_domain_nonce: self.clock_domain_nonce,
                            },
                            &bootstrap,
                            &self.credential.identity_fingerprint,
                        )
                        .await
                }
                AllocationRole::Worker => {
                    if self.launched_worker_leases.contains_key(&allocation.id) {
                        // A previous agent generation may still own this tree.
                        // An unchanged grant cannot be relaunched from list presence.
                        continue;
                    }
                    let bootstrap =
                        match self.control.worker_bootstrap(&heartbeat, allocation).await {
                            Ok(bootstrap) => bootstrap,
                            Err(runmat_execution_transport_native::TransportError::NotReady) => {
                                continue
                            }
                            Err(error) => return Err(error.into()),
                        };
                    validate_worker_bootstrap(
                        &self.credential,
                        allocation,
                        &bootstrap,
                        now_millis,
                    )?;
                    self.launched_worker_leases.insert(
                        allocation.id.clone(),
                        LaunchLedgerRecord {
                            fencing_token: allocation.fencing_token,
                            certified_acknowledged: false,
                        },
                    );
                    self.persist_launch_ledger()?;
                    self.processes
                        .launch_worker(
                            AllocationLaunchContext {
                                runmat_executable: &self.config.runmat_executable,
                                allocation,
                                sandbox: &sandbox,
                                server_url: &self.config.server_url,
                                clock_domain_nonce: self.clock_domain_nonce,
                            },
                            &bootstrap,
                            &inventory.host,
                            &spawn_evidence,
                        )
                        .await
                }
            };
            if let Err(error) = launch {
                if allocation.role == AllocationRole::Worker && !spawn_evidence.is_unverified() {
                    self.launched_worker_leases.remove(&allocation.id);
                    self.persist_launch_ledger()?;
                }
                if allocation.role == AllocationRole::Driver || !spawn_evidence.is_unverified() {
                    let _ = self.control.release(&heartbeat, allocation).await;
                }
                return Err(error);
            }
        }
        for allocation in allocations {
            if self.occupied_allocation_slots() >= self.config.maximum_allocations {
                break;
            }
            if allocation.state != "offered" || released_this_pass.contains(&allocation.id) {
                continue;
            }
            validate_offer(&allocation, &inventory, now_millis)?;
            let sandbox = prepare_sandbox(&self.config.state_directory, &allocation, &inventory)?;
            let evidence = prepare_endpoint_identity(
                &self.credential,
                &allocation,
                &sandbox,
                self.config.trust_tier,
                u64::try_from(now_millis).map_err(|_| {
                    AgentError::AllocationRejected("system clock predates Unix epoch".into())
                })?,
            )?;
            self.control
                .publish_endpoint_identity(&heartbeat, &allocation, evidence)
                .await?;
            self.control.accept(&heartbeat, &allocation).await?;
        }
        Ok(())
    }

    pub async fn rotate_credential(&mut self) -> AgentResult<()> {
        let heartbeat = self.heartbeat(inventory::collect(self.config.trust_tier)?)?;
        crate::enrollment::rotate(
            Arc::clone(&self.control),
            &mut self.store,
            &mut self.credential,
            heartbeat,
        )
        .await
    }

    fn heartbeat(
        &self,
        inventory: runmat_execution_transport_native::control::NodeInventory,
    ) -> AgentResult<NodeHeartbeat> {
        Ok(super::heartbeat_for(
            &self.credential,
            inventory,
            self.config.heartbeat_ttl.as_secs(),
        ))
    }

    async fn finish_drain(
        &mut self,
        heartbeat: &NodeHeartbeat,
        open_observers: u32,
    ) -> AgentResult<()> {
        if open_observers == 0
            && self
                .drain
                .complete_if_idle(self.occupied_allocation_slots())
        {
            self.control.complete_drain(heartbeat).await?;
        }
        Ok(())
    }

    fn record_retired(
        &mut self,
        retired: Vec<(String, Option<WorkerRetirementScope>)>,
    ) -> AgentResult<()> {
        if retired.is_empty() {
            return Ok(());
        }
        let signer =
            EndpointIdentitySigner::from_secret(decode_secret(&self.credential.identity_secret)?)
                .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
        let mut next = self.pending_release.clone();
        let mut committed = Vec::with_capacity(retired.len());
        for (allocation_id, scope) in retired {
            let mut original_scope = scope.clone();
            if let Some(original_scope) = &mut original_scope {
                original_scope.retired_monotonic = None;
                original_scope.retired_at_unix_millis = 0;
            }
            match next.get(&allocation_id) {
                Some(PendingRetirementRecord::Retiring { scope: retained })
                    if retained == &original_scope => {}
                _ => {
                    return Err(AgentError::AllocationRejected(
                        "retired process differs from its durable original owner".into(),
                    ));
                }
            }
            let evidence = if let Some(scope) = scope {
                let encrypted_attempt_receipts = retired_encrypted_receipts(&scope)?;
                let retired = scope.retired_monotonic.ok_or_else(|| {
                    AgentError::AllocationRejected(
                        "retired worker lacks node clock observation".into(),
                    )
                })?;
                let mut evidence = WorkerTreeRetirementEvidenceV1 {
                    schema_version: 1,
                    org_id: self.credential.org_id.clone(),
                    cluster_id: self.credential.cluster_id.clone(),
                    node_id: self.credential.node_id.clone(),
                    run_id: scope.run_id,
                    allocation_lease_id: scope.allocation_lease_id,
                    allocation_fencing_token: scope.allocation_fencing_token,
                    driver_fencing_token: scope.driver_fencing_token,
                    process_id: scope.process_id,
                    process_launch_nonce: scope.process_launch_nonce,
                    retired_at_unix_millis: scope.retired_at_unix_millis,
                    clock_domain_nonce: scope.clock_domain_nonce,
                    retired_monotonic_ticks: retired.ticks,
                    ticks_per_second: retired.ticks_per_second,
                    encrypted_attempt_receipts,
                    identity_public_key: self.credential.identity_public_key.clone(),
                    identity_fingerprint: self.credential.identity_fingerprint.clone(),
                    signature: Vec::new(),
                };
                signer
                    .sign_worker_tree_retirement(&mut evidence)
                    .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
                Some(evidence)
            } else {
                None
            };
            committed.push(allocation_id.clone());
            next.insert(
                allocation_id,
                PendingRetirementRecord::Certified { evidence },
            );
        }
        retirement::persist(
            &self.config.state_directory,
            self.config.maximum_allocations,
            &next,
        )?;
        self.pending_release = next;
        self.processes.commit_retired(&committed)?;
        Ok(())
    }

    fn persist_pending_release(&self) -> AgentResult<()> {
        retirement::persist(
            &self.config.state_directory,
            self.config.maximum_allocations,
            &self.pending_release,
        )
    }

    fn occupied_allocation_slots(&self) -> usize {
        let retained = self
            .launched_worker_leases
            .keys()
            .filter(|id| !self.processes.contains(id))
            .count();
        let pending_other = self
            .pending_release
            .keys()
            .filter(|id| {
                !self.processes.contains(id) && !self.launched_worker_leases.contains_key(*id)
            })
            .count();
        self.processes.active_count() + retained + pending_other
    }

    fn persist_launch_ledger(&self) -> AgentResult<()> {
        let path = self
            .config
            .state_directory
            .join("launched-worker-leases.json");
        let temporary = self.config.state_directory.join(format!(
            ".launched-worker-leases-{}.tmp",
            rand::random::<u64>()
        ));
        let bytes = serde_json::to_vec(&self.launched_worker_leases)?;
        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt as _;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        #[cfg(not(unix))]
        {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        runmat_process_host::durable_file::publish_synced_file(&temporary, &path)?;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        let deadline = tokio::time::Instant::now() + self.config.drain_timeout;
        while self.processes.active_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let state_directory = &self.config.state_directory;
        let maximum_allocations = self.config.maximum_allocations;
        let pending_release = &mut self.pending_release;
        let mut prepare = |allocation_id: &str, scope: Option<&WorkerRetirementScope>| {
            retirement::prepare(
                state_directory,
                maximum_allocations,
                pending_release,
                allocation_id,
                scope,
            )
        };
        let retired = self.processes.terminate_all(&mut prepare).await?;
        self.record_retired(retired)?;
        self.drain.begin();
        Ok(())
    }
}

fn cleanup_retired_attempt_journal(
    state_directory: &std::path::Path,
    allocation_id: &str,
) -> AgentResult<()> {
    if !allocation_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(AgentError::AllocationRejected(
            "retired allocation id is unsafe for journal cleanup".into(),
        ));
    }
    let path = state_directory
        .join("allocations")
        .join(allocation_id)
        .join("attempt-journal");
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_dir() {
        return Err(AgentError::AllocationRejected(
            "retired attempt journal is not an owned directory".into(),
        ));
    }
    for entry in std::fs::read_dir(&path)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err(AgentError::AllocationRejected(
                "retired attempt journal contains a non-file".into(),
            ));
        }
        std::fs::remove_file(entry.path())?;
    }
    std::fs::remove_dir(path)?;
    Ok(())
}

fn load_launch_ledger(
    state_directory: &std::path::Path,
) -> AgentResult<BTreeMap<String, LaunchLedgerRecord>> {
    let path = state_directory.join("launched-worker-leases.json");
    match std::fs::read(path) {
        Ok(bytes) => {
            if bytes.len() > 1024 * 1024 {
                return Err(AgentError::AllocationRejected(
                    "launch ledger exceeds bounded capacity".into(),
                ));
            }
            Ok(serde_json::from_slice(&bytes)?)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(AgentError::Io(error)),
    }
}

fn validate_worker_bootstrap(
    credential: &NodeCredential,
    allocation: &runmat_execution_transport_native::control::NodeAllocation,
    bootstrap: &runmat_execution_transport_native::control::WorkerBootstrapCredential,
    now_millis: i64,
) -> AgentResult<()> {
    if allocation.role != AllocationRole::Worker
        || bootstrap.org_id != credential.org_id
        || bootstrap.run_id != allocation.run_id
        || bootstrap.project_id != allocation.project_id
        || bootstrap.allocation_lease_id != allocation.id
        || bootstrap.allocation_fencing_token != allocation.fencing_token
        || bootstrap.driver_fencing_token == 0
        || bootstrap.endpoint_fingerprint.len() != 64
        || bootstrap.run_key_envelope.is_empty()
        || bootstrap.relay_path.is_empty()
        || bootstrap.relay_protocol != "runmat-worker-relay-v1"
        || bootstrap.relay_ticket.is_empty()
        || bootstrap.expires_at_millis <= now_millis
    {
        return Err(AgentError::AllocationRejected(
            "worker bootstrap authority does not match its allocation".into(),
        ));
    }
    Ok(())
}

fn validate_driver_bootstrap(
    credential: &NodeCredential,
    allocation: &runmat_execution_transport_native::control::NodeAllocation,
    bootstrap: &runmat_execution_transport_native::control::DriverBootstrapCredential,
    now_millis: i64,
) -> AgentResult<()> {
    if bootstrap.org_id != credential.org_id
        || bootstrap.run_id != allocation.run_id
        || bootstrap.project_id != allocation.project_id
        || bootstrap.allocation_lease_id != allocation.id
        || bootstrap.fencing_token == 0
        || bootstrap.credential.is_empty()
        || bootstrap.credential.len() > 256
        || bootstrap.expires_at_millis <= now_millis
    {
        return Err(AgentError::AllocationRejected(
            "driver bootstrap authority does not match its allocation".into(),
        ));
    }
    Ok(())
}
