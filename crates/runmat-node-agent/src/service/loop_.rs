use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::Utc;
use runmat_execution_transport_native::control::{
    AllocationRole, NodeControlPlane, NodeHeartbeat, ReconnectBackoff,
};

use crate::allocation::{
    prepare, prepare_endpoint_identity, validate_active, validate_offer, AllocationProcesses,
    DrainState,
};
use crate::enrollment::{CredentialStore, NodeCredential};
use crate::{inventory, AgentConfig, AgentError, AgentResult};

pub struct NodeAgentService {
    config: AgentConfig,
    control: Arc<dyn NodeControlPlane>,
    store: CredentialStore,
    credential: NodeCredential,
    processes: AllocationProcesses,
    drain: DrainState,
    pending_release: BTreeSet<String>,
}

impl NodeAgentService {
    pub fn load(config: AgentConfig, control: Arc<dyn NodeControlPlane>) -> AgentResult<Self> {
        config.validate()?;
        let store = CredentialStore::new(&config.state_directory);
        let credential = store.load()?;
        Ok(Self {
            config,
            control,
            store,
            credential,
            processes: AllocationProcesses::default(),
            drain: DrainState::Accepting,
            pending_release: BTreeSet::new(),
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
                    self.pending_release.extend(
                        self.processes
                            .enforce_local_limits(Utc::now().timestamp_millis())
                            .await?
                    );
                }
            }
        }
        self.shutdown().await
    }

    pub async fn reconcile_once(&mut self) -> AgentResult<()> {
        let inventory = inventory::collect()?;
        let heartbeat = self.heartbeat(inventory.clone())?;
        let status = self.control.heartbeat(heartbeat.clone()).await?;
        if status.credential_epoch != self.credential.credential_epoch {
            return Err(runmat_execution_transport_native::TransportError::StaleAuthority.into());
        }
        if self.credential.lease_epoch != status.lease_epoch {
            self.credential.lease_epoch = status.lease_epoch;
            self.store.store(&self.credential)?;
        }
        if status.state == "draining" {
            self.drain.begin();
        } else if status.state != "active" {
            self.processes.terminate_all().await?;
            return Err(AgentError::AllocationRejected(format!(
                "node state {} cannot run allocations",
                status.state
            )));
        }
        let allocations = self.control.allocations(&heartbeat).await?;
        let now_millis = Utc::now().timestamp_millis();
        self.processes.fence_stale(&allocations, now_millis).await?;
        let mut released_this_pass = BTreeSet::new();
        let pending = self.pending_release.iter().cloned().collect::<Vec<_>>();
        for allocation_id in pending {
            if let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.id == allocation_id)
            {
                self.control.release(&heartbeat, allocation).await?;
                released_this_pass.insert(allocation_id.clone());
            }
            self.pending_release.remove(&allocation_id);
        }
        for allocation_id in self.processes.reap_finished()? {
            if let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.id == allocation_id)
            {
                self.control.release(&heartbeat, allocation).await?;
                released_this_pass.insert(allocation_id);
            }
        }
        if self.drain != DrainState::Accepting {
            return self.finish_drain(&heartbeat).await;
        }
        for allocation in &allocations {
            if self.processes.active_count() >= self.config.maximum_allocations {
                break;
            }
            if allocation.state != "active"
                || self.processes.contains(&allocation.id)
                || released_this_pass.contains(&allocation.id)
            {
                continue;
            }
            validate_active(allocation, &inventory, now_millis)?;
            let sandbox = prepare(&self.config.state_directory, allocation, &inventory)?;
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
                            &self.config.runmat_executable,
                            allocation,
                            &sandbox,
                            &self.config.server_url,
                            &bootstrap,
                        )
                        .await
                }
                AllocationRole::Worker => {
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
                    self.processes
                        .launch_worker(
                            &self.config.runmat_executable,
                            allocation,
                            &sandbox,
                            &self.config.server_url,
                            &bootstrap,
                        )
                        .await
                }
            };
            if let Err(error) = launch {
                let _ = self.control.release(&heartbeat, allocation).await;
                return Err(error);
            }
        }
        for allocation in allocations {
            if self.processes.active_count() >= self.config.maximum_allocations {
                break;
            }
            if allocation.state != "offered" || released_this_pass.contains(&allocation.id) {
                continue;
            }
            validate_offer(&allocation, &inventory, now_millis)?;
            let sandbox = prepare(&self.config.state_directory, &allocation, &inventory)?;
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
        let heartbeat = self.heartbeat(inventory::collect()?)?;
        crate::enrollment::rotate(
            Arc::clone(&self.control),
            &self.store,
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

    async fn finish_drain(&mut self, heartbeat: &NodeHeartbeat) -> AgentResult<()> {
        if self.drain.complete_if_idle(self.processes.active_count()) {
            self.control.complete_drain(heartbeat).await?;
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        let deadline = tokio::time::Instant::now() + self.config.drain_timeout;
        while self.processes.active_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        self.processes.terminate_all().await?;
        self.drain.begin();
        Ok(())
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
