use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::Utc;
use runmat_execution_transport_native::control::{
    NodeControlPlane, NodeHeartbeat, ReconnectBackoff,
};

use crate::allocation::{prepare, validate_offer, AllocationProcesses, DrainState};
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
        let pending = self.pending_release.iter().cloned().collect::<Vec<_>>();
        for allocation_id in pending {
            if let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.id == allocation_id)
            {
                self.control.release(&heartbeat, allocation).await?;
            }
            self.pending_release.remove(&allocation_id);
        }
        for allocation_id in self.processes.reap_finished()? {
            if let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.id == allocation_id)
            {
                self.control.release(&heartbeat, allocation).await?;
            }
        }
        if self.drain != DrainState::Accepting {
            return self.finish_drain(&heartbeat).await;
        }
        for allocation in allocations {
            if self.processes.active_count() >= self.config.maximum_allocations {
                break;
            }
            if allocation.state != "offered" {
                continue;
            }
            validate_offer(&allocation, &inventory, now_millis)?;
            let sandbox = prepare(&self.config.state_directory, &allocation, &inventory)?;
            self.control.accept(&heartbeat, &allocation).await?;
            if let Err(error) = self
                .processes
                .launch_driver(&self.config.runmat_executable, &allocation, &sandbox)
                .await
            {
                let _ = self.control.release(&heartbeat, &allocation).await;
                return Err(error);
            }
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
