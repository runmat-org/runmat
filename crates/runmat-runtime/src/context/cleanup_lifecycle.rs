//! Explicit, non-cloneable host authority for session cleanup callbacks.
//! RuntimeContext remains a cloneable service handle, never a host lease.

use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::{LocalTaskCommandOrigin, LocalTaskOwnerToken, RuntimeContext, RuntimeContextState};
use crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupRegistry;
use crate::RuntimeError;

#[path = "cleanup_lifecycle/wake.rs"]
mod wake;
use wake::LifecycleWake;

#[cfg(not(target_arch = "wasm32"))]
type LocalSupervisor = super::RuntimeLocalTaskSupervisor;
#[cfg(target_arch = "wasm32")]
type LocalSupervisor = super::BrowserLocalTaskSupervisor;

/// The actual Core/AOT/browser/standalone execution host retains this owner.
/// The context and its callback tasks only retain weak service/registry ports.
pub struct RuntimeLifecycleHost {
    supervisor: Rc<LocalSupervisor>,
    registry: Rc<CleanupRegistry>,
    state: Weak<RuntimeContextState>,
    owner_token: LocalTaskOwnerToken,
    wake: Arc<LifecycleWake>,
    _root_observer: runmat_gc::RootRetirementObserver,
    _dispatcher: super::RuntimeLocalTaskLease,
    closed: Cell<bool>,
}

impl RuntimeLifecycleHost {
    /// Install only after the same host supervisor's weak submission port has
    /// been installed in the selected RuntimeContext. No callback is admitted
    /// to an unhosted bare service context.
    pub fn new(
        runtime: &RuntimeContext,
        supervisor: Rc<LocalSupervisor>,
    ) -> Result<Self, RuntimeError> {
        let Some(tasks) = runtime.service_ports().local_tasks() else {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostUnavailable",
                "onCleanup lifecycle host requires a local task submission port",
            ));
        };
        let Some(owner_token) = tasks.owner_token() else {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostForeignPort",
                "onCleanup lifecycle host requires an exact task supervisor port",
            ));
        };
        if !owner_token.same(&supervisor.owner_token()) {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostForeignPort",
                "onCleanup lifecycle host and local task port belong to different sessions",
            ));
        }
        let mut slot = runtime
            .state()
            .cleanup_registry
            .try_borrow_mut()
            .map_err(|_| {
                crate::runtime_error::semantic_error(
                    "RunMat:onCleanup:LifecycleHostBusy",
                    "onCleanup lifecycle host registration is reentrant",
                )
            })?;
        if slot.as_ref().and_then(Weak::upgrade).is_some() {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostAlreadyInstalled",
                "onCleanup lifecycle host already owns this session",
            ));
        }
        let registry = Rc::new(CleanupRegistry::new());
        let wake = Arc::new(LifecycleWake::new());
        let weak_wake = Arc::downgrade(&wake);
        let observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |_| {
            if let Some(wake) = weak_wake.upgrade() {
                wake.notify();
            }
        }))
        .ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostExhausted",
                "onCleanup lifecycle host observer identity is exhausted",
            )
        })?;
        let dispatcher_registry = Rc::downgrade(&registry);
        let dispatcher_wake = Arc::clone(&wake);
        let dispatcher = runtime
            .spawn_local_with_command_origin(
                "onCleanup lifecycle dispatcher",
                LocalTaskCommandOrigin::Unknown,
                move |dispatcher_runtime| {
                    Box::pin(async move {
                        while dispatcher_wake.next().await {
                            let Some(registry) = dispatcher_registry.upgrade() else {
                                break;
                            };
                            // The host starts before a program can install its
                            // callable projection. Select its own live session
                            // owner at this wake, not a construction-time
                            // snapshot or an unrelated foreground projection.
                            let _selected = super::RuntimeContextGuard::enter(
                                dispatcher_runtime.clone(),
                            );
                            crate::builtins::introspection::on_cleanup::lifecycle::dispatch::schedule_eligible(
                                &dispatcher_runtime,
                                &registry,
                                &[],
                            );
                        }
                    })
                },
            )
            .map_err(|error| {
                crate::runtime_error::semantic_error(
                    "RunMat:onCleanup:LifecycleHostUnavailable",
                    format!("onCleanup lifecycle dispatcher could not start: {error}"),
                )
            })?;
        *slot = Some(Rc::downgrade(&registry));
        Ok(Self {
            supervisor,
            registry,
            state: Rc::downgrade(runtime.state()),
            owner_token,
            wake,
            _root_observer: observer,
            _dispatcher: dispatcher,
            closed: Cell::new(false),
        })
    }

    /// Forced host disposal is synchronous: queued/running callbacks receive
    /// an explicit Abandoned receipt, never a fabricated success or replay.
    pub fn force_close(&self) {
        if self.closed.replace(true) {
            return;
        }
        self.registry.force_abandon();
        self.wake.close();
        self.supervisor.shutdown();
        self.registry.release_abandoned_after_task_shutdown();
        if let Some(state) = self.state.upgrade() {
            if let Ok(mut slot) = state.cleanup_registry.try_borrow_mut() {
                slot.take();
            }
        }
    }

    /// Worker invocation retirement requires actual child-frame destruction,
    /// not merely the synchronous cancellation outcomes used by ordinary close.
    #[cfg(target_arch = "wasm32")]
    pub async fn force_close_and_retire(&self) {
        self.force_close();
        self.supervisor.shutdown_and_retire().await;
    }

    fn ensure_runtime(&self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        let matching = self
            .state
            .upgrade()
            .is_some_and(|state| Rc::ptr_eq(&state, runtime.state()));
        let matching_port = runtime
            .service_ports()
            .local_tasks()
            .and_then(|tasks| tasks.owner_token())
            .is_some_and(|token| token.same(&self.owner_token));
        if matching && matching_port && !self.closed.get() {
            Ok(())
        } else {
            Err(crate::runtime_error::semantic_error(
                "RunMat:onCleanup:LifecycleHostForeignContext",
                "cleanup lifecycle host does not own this runtime context",
            ))
        }
    }

    /// Drive admitted callbacks on the existing host executor. Externally
    /// owned Active objects remain registered and are never awaited here.
    /// Callers retire their owned workspace/frame roots before graceful close.
    pub async fn drain_gracefully(&self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        self.ensure_runtime(runtime)?;
        crate::builtins::introspection::on_cleanup::lifecycle::dispatch::drain_eligible(
            runtime,
            &self.registry,
            &[],
        )
        .await
    }

    /// Normal close retires this host's exact workspace bindings first. The
    /// detached root remains alive (but excluded by registration ID from the
    /// last-owner proof) until every admitted callback has settled. A callback
    /// can publish new session-owned bindings; retire those in subsequent
    /// batches before declaring the still-open workspace quiescent.
    pub async fn retire_workspace_and_drain(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.ensure_runtime(runtime)?;
        runtime
            .clone()
            .scope(async {
                let mut detached = crate::workspace::try_detach_selected(
                    &crate::workspace::WorkspaceDetachSelection::All,
                )?;
                let mut result = Ok(());
                loop {
                    result = crate::runtime_error::complete_with_cleanup(
                        result,
                        crate::builtins::introspection::on_cleanup::lifecycle::dispatch::drain_eligible(
                            runtime,
                            &self.registry,
                            &[detached.root_registration()],
                        )
                        .await,
                    );
                    drop(detached);
                    result = crate::runtime_error::complete_with_cleanup(
                        result,
                        self.drain_gracefully(runtime).await,
                    );
                    detached = match crate::workspace::try_detach_selected(
                        &crate::workspace::WorkspaceDetachSelection::All,
                    ) {
                        Ok(detached) => detached,
                        Err(error) => {
                            return crate::runtime_error::complete_with_cleanup(result, Err(error));
                        }
                    };
                    // Only newly session-owned bindings repeat this loop.
                    // Externally owned Active objects never keep close alive.
                    // No callback is awaited after this empty-batch proof.
                    if detached.is_empty() {
                        drop(detached);
                        return result;
                    }
                }
            })
            .await
    }

    /// Session safepoint: inspect at most the current rooted graph and submit
    /// only proven-eligible work; idle active objects never get waiting tasks.
    pub fn service_safepoint(&self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        self.ensure_runtime(runtime)?;
        crate::builtins::introspection::on_cleanup::lifecycle::dispatch::schedule_eligible(
            runtime,
            &self.registry,
            &[],
        );
        Ok(())
    }
}

impl Drop for RuntimeLifecycleHost {
    fn drop(&mut self) {
        self.force_close();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::context::{RuntimeLocalTaskSupervisor, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;

    #[test]
    fn same_state_with_foreign_task_port_cannot_drive_host() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let owner = Rc::new(RuntimeLocalTaskSupervisor::new());
        let local_identity = runtime.local_identity();
        let runtime = runtime.with_service_ports(
            RuntimeServicePorts::default().with_local_tasks(owner.submission_port(local_identity)),
        );
        let host = RuntimeLifecycleHost::new(&runtime, owner).expect("matching owner");

        let foreign_owner = Rc::new(RuntimeLocalTaskSupervisor::new());
        let foreign = runtime.clone().with_service_ports(
            RuntimeServicePorts::default()
                .with_local_tasks(foreign_owner.submission_port(runtime.local_identity())),
        );
        let error = host
            .service_safepoint(&foreign)
            .expect_err("same state does not confer foreign task authority");
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:LifecycleHostForeignContext")
        );
        host.service_safepoint(&runtime)
            .expect("original host port");
    }
}
