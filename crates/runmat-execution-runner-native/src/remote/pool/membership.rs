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
        let described = channel.describe_worker().await?;
        if described != worker {
            return Err(NativeExecutionError::Protocol(
                "remote worker inventory differs from its encrypted description".into(),
            ));
        }
        let node = channel.node_identity().to_string();
        let node_was_installed = self.installed_nodes.lock().await.contains(&node);
        let receipt = if node_was_installed {
            match channel.activate_bundle(self.bundle_digest).await {
                Ok(receipt) => receipt,
                Err(_) => {
                    channel
                        .install_bundle(self.bundle_digest, &self.bundle)
                        .await?
                }
            }
        } else {
            channel
                .install_bundle(self.bundle_digest, &self.bundle)
                .await?
        };
        if receipt.bundle_digest != self.bundle_digest
            || receipt.bundle_identity != self.bundle_identity
            || receipt.project_revision != self.project_revision
            || receipt.stored_bytes != self.bundle.len() as u64
        {
            return Err(NativeExecutionError::Protocol(
                "remote worker did not acknowledge the exact bundle".into(),
            ));
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
            driver.handle(DriverCommand::RegisterWorker(worker))?
        };
        if !node_was_installed {
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
        let (channel, actions) = if lost {
            let mut channels = self.channels.write().await;
            let channel = channels.remove(&worker_id).map(|route| route.channel);
            let actions = self
                .driver
                .lock()
                .expect("remote driver poisoned")
                .handle(DriverCommand::WorkerLost(worker_id))?;
            drop(channels);
            (channel, actions)
        } else {
            let channel = self
                .channels
                .read()
                .await
                .get(&worker_id)
                .map(|route| Arc::clone(&route.channel));
            let actions = self
                .driver
                .lock()
                .expect("remote driver poisoned")
                .handle(DriverCommand::DrainWorker(worker_id))?;
            (channel, actions)
        };
        self.dispatch(actions);
        self.resolve_non_success_terminals();
        if !lost {
            if let Some(channel) = channel {
                if let Err(error) = channel.drain().await {
                    let mut channels = self.channels.write().await;
                    channels.remove(&worker_id);
                    let actions = self
                        .driver
                        .lock()
                        .expect("remote driver poisoned")
                        .handle(DriverCommand::WorkerLost(worker_id))?;
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
