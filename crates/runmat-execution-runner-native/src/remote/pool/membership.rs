//! Worker membership, routes and shared-pool availability transitions.
use super::*;

impl RemotePoolDriver {
    pub async fn add_worker(
        self: &Arc<Self>,
        channel: Arc<dyn RemoteWorkerChannel>,
    ) -> NativeExecutionResult<()> {
        let _registration = self.worker_registration.lock().await;
        let worker = channel.worker().clone();
        if worker.pool_id != self.pool_id {
            return Err(NativeExecutionError::Configuration(
                "remote worker belongs to a different pool".into(),
            ));
        }
        {
            let driver = self.driver.lock().expect("remote driver poisoned");
            let snapshot = driver.snapshot();
            let pool = snapshot
                .pools
                .get(&self.pool_id)
                .ok_or_else(|| NativeExecutionError::Protocol("remote pool disappeared".into()))?;
            let recoverable_lost_worker = pool.workers.get(&worker.id).is_some_and(|existing| {
                existing.lifecycle == runmat_execution_runner::pool::WorkerLifecycle::Lost
                    && existing.active_attempts == 0
                    && existing.spec == worker
            });
            if pool.workers.contains_key(&worker.id) && !recoverable_lost_worker {
                return Err(NativeExecutionError::Configuration(format!(
                    "remote worker {} is already registered",
                    worker.id
                )));
            }
            let live_worker_count = pool
                .workers
                .values()
                .filter(|worker| {
                    !matches!(
                        worker.lifecycle,
                        runmat_execution_runner::pool::WorkerLifecycle::Lost
                            | runmat_execution_runner::pool::WorkerLifecycle::Stopped
                    )
                })
                .count();
            if live_worker_count >= pool.spec.max_workers as usize && !recoverable_lost_worker {
                return Err(NativeExecutionError::Configuration(format!(
                    "remote pool {} is at its worker limit",
                    self.pool_id
                )));
            }
        }
        let mut described = channel.describe_worker().await?;
        // Software support is admitted only from the route's authenticated
        // ReadyV1, never from the static description or allocation inventory.
        for capability in super::super::worker_protocol::software_capabilities() {
            described.resources.capabilities.remove(&capability);
        }
        if described != worker {
            return Err(NativeExecutionError::Protocol(
                "remote worker inventory differs from its encrypted description".into(),
            ));
        }
        let node = channel.node_identity().to_string();
        let node_was_installed = self.installed_nodes.lock().await.contains(&node);
        if let Some(bundle) = &self.direct_bundle {
            let receipt = if node_was_installed {
                match channel.activate_bundle(bundle.digest).await {
                    Ok(receipt) => receipt,
                    Err(_) => channel.install_bundle(bundle.digest, &bundle.bytes).await?,
                }
            } else {
                channel.install_bundle(bundle.digest, &bundle.bytes).await?
            };
            if receipt.bundle_digest != bundle.digest
                || receipt.bundle_identity != bundle.identity
                || receipt.project_revision != bundle.project_revision
                || receipt.stored_bytes != bundle.bytes.len() as u64
            {
                return Err(NativeExecutionError::Protocol(
                    "remote worker did not acknowledge the exact bundle".into(),
                ));
            }
        }
        // Acquire every fallible asynchronous guard before mutating driver state.
        // Once RegisterWorker succeeds, publishing the route and installed-node
        // bookkeeping is synchronous, so cancellation cannot leave a half-admitted
        // worker behind.
        let mut installed_nodes = self.installed_nodes.lock().await;
        let mut channels = self.channels.write().await;
        let route_generation = self
            .next_route_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1)
            })
            .map_err(|_| {
                NativeExecutionError::Protocol("remote route generation exhausted".into())
            })?;
        let actions = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let mut actions = driver.handle(DriverCommand::RegisterWorker(worker))?;
            actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
                pool_id: self.pool_id,
            })?);
            actions
        };
        if !node_was_installed && self.direct_bundle.is_some() {
            installed_nodes.insert(node);
        }
        channels.insert(
            channel.worker().id,
            WorkerRoute {
                identity: WorkerRouteIdentity {
                    worker_id: channel.worker().id,
                    generation: route_generation,
                },
                channel,
            },
        );
        drop(channels);
        drop(installed_nodes);
        self.dispatch(actions);
        self.refresh_pool_state()?;
        self.refresh_worker_software_capabilities().await?;
        Ok(())
    }

    /// Pull only readiness already authenticated by each encrypted route.
    /// This does not query or wait for a worker and is safe on the ordinary
    /// V10 reconciliation path.
    pub async fn refresh_worker_software_capabilities(
        self: &Arc<Self>,
    ) -> NativeExecutionResult<()> {
        let observed = self
            .channels
            .read()
            .await
            .iter()
            .filter_map(|(id, route)| {
                route
                    .channel
                    .verified_software_capabilities()
                    .map(|capabilities| (*id, capabilities))
            })
            .collect::<Vec<_>>();
        if observed.is_empty() {
            return Ok(());
        }
        let actions = {
            let mut applied = self
                .applied_software_readiness
                .lock()
                .expect("worker readiness registry poisoned");
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let mut actions = Vec::new();
            for (worker_id, capabilities) in observed {
                if applied.contains(&worker_id) {
                    continue;
                }
                actions.extend(
                    driver.handle(DriverCommand::AdmitWorkerSoftwareCapabilities {
                        worker_id,
                        capabilities,
                    })?,
                );
                actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
                    pool_id: self.pool_id,
                })?);
                applied.insert(worker_id);
            }
            actions
        };
        self.dispatch(actions);
        Ok(())
    }

    pub fn resize(self: &Arc<Self>, desired_workers: u32) -> NativeExecutionResult<()> {
        let actions = self.driver.lock().expect("remote driver poisoned").handle(
            DriverCommand::ResizePool {
                pool_id: self.pool_id,
                request: ResizeRequest { desired_workers },
            },
        )?;
        self.dispatch(actions);
        Ok(())
    }

    pub async fn remove_worker(
        self: &Arc<Self>,
        worker_id: runmat_execution::identity::WorkerId,
        lost: bool,
    ) -> NativeExecutionResult<()> {
        let _draining_registration = if lost {
            None
        } else {
            Some(self.worker_registration.lock().await)
        };
        self.applied_software_readiness
            .lock()
            .expect("worker readiness registry poisoned")
            .remove(&worker_id);
        let (channel, actions) = if lost {
            let mut channels = self.channels.write().await;
            let channel = channels.remove(&worker_id).map(|route| route.channel);
            let actions = self.worker_transition(DriverCommand::WorkerLost(worker_id))?;
            drop(channels);
            (channel, actions)
        } else {
            let channel = self
                .channels
                .read()
                .await
                .get(&worker_id)
                .map(|route| Arc::clone(&route.channel));
            let actions = self.worker_transition(DriverCommand::DrainWorker(worker_id))?;
            (channel, actions)
        };
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        if !lost {
            if let Some(channel) = channel {
                if let Err(error) = channel.drain().await {
                    let mut channels = self.channels.write().await;
                    channels.remove(&worker_id);
                    let actions = self.worker_transition(DriverCommand::WorkerLost(worker_id))?;
                    drop(channels);
                    self.dispatch(actions);
                    self.resolve_non_success_terminals();
                    self.refresh_pool_state()?;
                    return Err(error);
                }
            }
            self.channels.write().await.remove(&worker_id);
        }
        self.refresh_pool_state()?;
        Ok(())
    }

    /// Apply a membership transition and re-derive the pool's admitted capabilities.
    fn worker_transition(
        &self,
        command: DriverCommand,
    ) -> NativeExecutionResult<Vec<DriverAction>> {
        let mut driver = self.driver.lock().expect("remote driver poisoned");
        let mut actions = driver.handle(command)?;
        actions.extend(driver.handle(DriverCommand::RefreshPoolCapabilities {
            pool_id: self.pool_id,
        })?);
        Ok(actions)
    }

    pub(super) fn refresh_pool_state(self: &Arc<Self>) -> NativeExecutionResult<()> {
        let mut driver = self.driver.lock().expect("remote driver poisoned");
        let pool = driver.pool_record(self.pool_id)?;
        let target = pool.worker_availability_state();
        if pool.state != target {
            let actions = driver.handle(DriverCommand::SetPoolState {
                pool_id: self.pool_id,
                state: target,
            })?;
            drop(driver);
            self.dispatch(actions);
        }
        Ok(())
    }
}
