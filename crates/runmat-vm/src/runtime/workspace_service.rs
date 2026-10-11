use std::cell::{Cell, RefCell};
use std::rc::Rc;

use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::workspace::workspace_frame_for_service;
use super::workspace::{
    reconcile_workspace_frame, release_pending_workspace_service_frame,
    reserve_workspace_service_frame, workspace_frame_assign, workspace_frame_assign_from_executor,
    workspace_frame_clear, workspace_frame_declare_global, workspace_frame_detach_selected,
    workspace_frame_lookup, workspace_frame_name_is_global, workspace_frame_publishes,
    workspace_frame_remove, workspace_frame_snapshot, PendingWorkspaceAdmission, WorkspaceFrameId,
    WorkspaceFrameLifecycle,
};

/// Exact current-frame service installed around one interpreter invocation.
/// Top-level frames delegate admitted mutations to the Runtime projection;
/// procedure and transactional dynamic-eval frames remain isolated.
pub(crate) struct VmWorkspaceService {
    delegate: Rc<RefCell<Option<Rc<dyn RuntimeWorkspaceService>>>>,
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    frame: WorkspaceFrameId,
    lifecycle: Rc<Cell<WorkspaceFrameLifecycle>>,
    delegate_revision: Cell<Option<runmat_runtime::workspace::WorkspaceRevision>>,
    session_variables: runmat_runtime::workspace::session::SessionVariableAuthority,
}

impl VmWorkspaceService {
    pub(crate) fn install(
        runtime: runmat_runtime::context::RuntimeContext,
        admission: Option<PendingWorkspaceAdmission>,
    ) -> (
        runmat_runtime::context::RuntimeContext,
        super::workspace::WorkspaceFrameActivation,
    ) {
        let runtime_id = runtime.local_identity();
        let (runtime, service) = Self::prepare(runtime, admission);
        (
            runtime,
            super::workspace::frame_activation(runtime_id, service.frame),
        )
    }

    fn prepare(
        runtime: runmat_runtime::context::RuntimeContext,
        admission: Option<PendingWorkspaceAdmission>,
    ) -> (
        runmat_runtime::context::RuntimeContext,
        Rc<VmWorkspaceService>,
    ) {
        let delegate = runtime.service_ports().workspace().cloned();
        let runtime_id = runtime.local_identity();
        let pending =
            reserve_workspace_service_frame(runtime_id, admission.map(|item| item.into_state()));
        *pending.delegate.borrow_mut() = delegate.clone();
        let session_variables = runtime.session_variable_authority();
        *pending.session_variables.borrow_mut() = Some(session_variables.clone());
        let lineage = delegate
            .as_ref()
            .and_then(|delegate| workspace_frame_for_service(runtime_id, delegate));
        pending.parent.set(lineage.map(|(parent, _)| parent));
        pending.base.set(lineage.map(|(_, base)| base));
        let service = Rc::new(Self {
            delegate: pending.delegate,
            runtime: runtime_id,
            frame: pending.id,
            lifecycle: pending.lifecycle,
            delegate_revision: Cell::new(None),
            session_variables,
        });
        let port: Rc<dyn RuntimeWorkspaceService> = service.clone();
        *pending.service.borrow_mut() = Some(Rc::downgrade(&port));
        let ports = runtime.service_ports().clone().with_workspace(port);
        (runtime.with_service_ports(ports), service)
    }

    fn should_publish(&self) -> bool {
        workspace_frame_publishes(self.frame).unwrap_or(false)
    }

    fn is_pending(&self) -> bool {
        self.lifecycle.get() == WorkspaceFrameLifecycle::Pending
    }

    fn ensure_frame_active(&self) -> Result<(), RuntimeError> {
        match self.lifecycle.get() {
            WorkspaceFrameLifecycle::Active => Ok(()),
            WorkspaceFrameLifecycle::Pending => Err(frame_error(
                "RunMat:Workspace:FramePending",
                "workspace frame has not entered execution",
            )),
            WorkspaceFrameLifecycle::Closed => Err(frame_error(
                "RunMat:Workspace:FrameClosed",
                "workspace frame is closed",
            )),
        }
    }

    fn delegate(&self) -> Option<Rc<dyn RuntimeWorkspaceService>> {
        self.delegate.borrow().clone()
    }
}

impl Drop for VmWorkspaceService {
    fn drop(&mut self) {
        if self.lifecycle.get() == WorkspaceFrameLifecycle::Pending {
            release_pending_workspace_service_frame(self.runtime, self.frame);
            self.lifecycle.set(WorkspaceFrameLifecycle::Closed);
            let retired_delegate = self.delegate.borrow_mut().take();
            drop(retired_delegate);
        }
    }
}

impl RuntimeWorkspaceService for VmWorkspaceService {
    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        self.ensure_frame_active()?;
        self.ensure_effect_authority()?;
        let delegate = self.should_publish().then(|| self.delegate()).flatten();
        workspace_frame_detach_selected(self.frame, selection, |detached| {
            let _retired_delegate = if let Some(delegate) = delegate {
                let delegated = match selection {
                    WorkspaceDetachSelection::All => WorkspaceDetachSelection::All,
                    WorkspaceDetachSelection::Names(names) => {
                        WorkspaceDetachSelection::Names(names.clone())
                    }
                    WorkspaceDetachSelection::Except(_) => WorkspaceDetachSelection::Names(
                        detached
                            .entries()
                            .iter()
                            .map(|(name, _)| name.clone())
                            .collect(),
                    ),
                };
                Some(delegate.detach_selected(&delegated)?)
            } else {
                None
            };
            Ok(())
        })
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        if self.is_pending() {
            return self
                .delegate()
                .and_then(|delegate| delegate.visit_retained_values(visitor));
        }
        let snapshot = workspace_frame_snapshot(self.frame)?;
        for (_, value) in snapshot {
            visitor(&value)?;
        }
        Some(())
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        match self.lifecycle.get() {
            WorkspaceFrameLifecycle::Pending => self.delegate()?.lookup(name),
            WorkspaceFrameLifecycle::Active => workspace_frame_lookup(self.frame, name),
            WorkspaceFrameLifecycle::Closed => None,
        }
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        match self.lifecycle.get() {
            WorkspaceFrameLifecycle::Pending => self
                .delegate()
                .map(|delegate| delegate.snapshot())
                .unwrap_or_default(),
            WorkspaceFrameLifecycle::Active => {
                workspace_frame_snapshot(self.frame).unwrap_or_default()
            }
            WorkspaceFrameLifecycle::Closed => Vec::new(),
        }
    }

    fn global_names(&self) -> Vec<String> {
        match self.lifecycle.get() {
            WorkspaceFrameLifecycle::Pending => self
                .delegate()
                .map(|delegate| delegate.global_names())
                .unwrap_or_default(),
            WorkspaceFrameLifecycle::Active => {
                self.session_variables.global_names().unwrap_or_default()
            }
            WorkspaceFrameLifecycle::Closed => Vec::new(),
        }
    }

    fn try_lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        if !self.is_pending() {
            self.ensure_frame_active()?;
            self.ensure_effect_authority()?;
            return Ok(workspace_frame_lookup(self.frame, name));
        }
        self.delegate()
            .map(|delegate| delegate.try_lookup(name))
            .unwrap_or(Ok(None))
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
        if self.is_pending() {
            return self
                .delegate()
                .map(|delegate| delegate.try_snapshot())
                .unwrap_or_else(|| Ok(Vec::new()));
        }
        self.ensure_frame_active()?;
        self.ensure_effect_authority()?;
        workspace_frame_snapshot(self.frame)
            .ok_or_else(|| frame_error("RunMat:Workspace:FrameClosed", "workspace frame is closed"))
    }

    fn try_global_names(&self) -> Result<Vec<String>, RuntimeError> {
        if !self.is_pending() {
            self.ensure_effect_authority()?;
        }
        self.session_variables.global_names()
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.ensure_frame_active()?;
        if super::workspace::frame_is_detaching(self.frame) {
            return Err(frame_error(
                "RunMat:Workspace:FrameUnavailable",
                "workspace frame has an active detach transaction",
            ));
        }
        if let Some(delegate) = self.delegate() {
            delegate.ensure_effect_authority()?;
            if self.should_publish() {
                let revision = delegate.workspace_revision()?;
                if revision.is_some() && revision == self.delegate_revision.get() {
                    return Ok(());
                }
                let canonical = delegate.try_snapshot()?;
                reconcile_workspace_frame(self.frame, canonical)
                    .map_err(workspace_operation_error)?;
                self.delegate_revision.set(revision);
            }
        }
        Ok(())
    }

    fn workspace_revision(
        &self,
    ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, RuntimeError> {
        self.ensure_effect_authority()?;
        Ok(self.delegate_revision.get())
    }

    fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
        self.ensure_frame_active()?;
        if super::workspace::frame_is_detaching(self.frame) {
            return Err(frame_error(
                "RunMat:Workspace:FrameUnavailable",
                "workspace frame has an active detach transaction",
            ));
        }
        if let Some(delegate) = self.delegate() {
            delegate.ensure_mutation_authority()?;
        }
        Ok(())
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()?;
        runmat_runtime::execution::validate_storable_value(&value)?;
        let global = workspace_frame_name_is_global(self.frame, name).unwrap_or(false);
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                if global {
                    delegate.remove(name)?;
                } else {
                    delegate.assign(name, value.clone())?;
                }
            }
        }
        if global {
            self.session_variables.store_global(name, value.clone())?;
        }
        workspace_frame_assign(self.frame, name, value).map_err(workspace_operation_error)
    }

    fn assign_executor_slot(
        &self,
        _slot: usize,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        self.ensure_mutation_authority()?;
        runmat_runtime::execution::validate_storable_value(&value)?;
        let global = workspace_frame_name_is_global(self.frame, name).unwrap_or(false);
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                if global {
                    delegate.remove(name)?;
                } else {
                    delegate.assign(name, value.clone())?;
                }
            }
        }
        if global {
            self.session_variables.store_global(name, value.clone())?;
        }
        workspace_frame_assign_from_executor(self.frame, name, value)
            .map_err(workspace_operation_error)
    }

    fn clear(&self) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                delegate.clear()?;
            }
        }
        workspace_frame_clear(self.frame).map_err(workspace_operation_error)
    }

    fn declare_global_binding(&self, slot: usize, name: &str) -> Result<(), RuntimeError> {
        self.ensure_mutation_authority()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                delegate.bind_base_global(name)?;
            }
        }
        workspace_frame_declare_global(self.frame, slot, name.to_string()).ok_or_else(|| {
            frame_error("RunMat:Workspace:FrameClosed", "workspace frame is closed")
        })?;
        let value = self.session_variables.declare_global(name)?;
        workspace_frame_assign_from_executor(self.frame, name, value)
            .map_err(workspace_operation_error)
    }

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()?;
        if self.should_publish() {
            if let Some(delegate) = self.delegate() {
                delegate.unbind_base_global(name)?;
            }
        }
        workspace_frame_remove(self.frame, name).map_err(workspace_operation_error)
    }
}

fn workspace_operation_error(message: String) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:Workspace:OperationFailed", message)
}

fn frame_error(identifier: &'static str, message: &'static str) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(identifier, message)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::{HashMap, HashSet};

    use runmat_runtime::context::RuntimeWorkspaceService;
    use runmat_runtime::workspace::bindings::FrameGlobalBindings;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    use super::*;

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn poll<T>(future: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        future.poll(&mut Context::from_waker(&waker))
    }

    fn workspace_scope<F: Future>(
        runtime: runmat_runtime::context::RuntimeContext,
        service: &Rc<VmWorkspaceService>,
        future: F,
    ) -> impl Future<Output = F::Output> {
        let activation =
            super::super::workspace::frame_activation(runtime.local_identity(), service.frame);
        activation.scope(runtime.scope(future))
    }

    struct CountingWorkspaceService {
        inner: Rc<dyn RuntimeWorkspaceService>,
        snapshots: Cell<usize>,
        mutation_admissions: Cell<usize>,
    }

    #[test]
    fn sibling_futures_share_a_runtime_without_crossing_frames_or_receipts() {
        fn sibling(
            runtime: runmat_runtime::context::RuntimeContext,
            service: Rc<VmWorkspaceService>,
            value: f64,
        ) -> Pin<Box<impl Future<Output = ()>>> {
            let activation =
                super::super::workspace::frame_activation(runtime.local_identity(), service.frame);
            Box::pin(activation.scope(runtime.scope(async move {
                let mut values = vec![Value::Num(value)];
                let _guard = super::super::workspace::set_workspace_state(
                    HashMap::from([("x".to_string(), 0)]),
                    HashSet::from(["x".to_string()]),
                    &mut values,
                    FrameGlobalBindings::default(),
                );
                let mut yielded = false;
                std::future::poll_fn(move |_| {
                    if yielded {
                        Poll::Ready(())
                    } else {
                        yielded = true;
                        Poll::Pending
                    }
                })
                .await;
                assert_eq!(
                    workspace_frame_lookup(service.frame, "x"),
                    Some(Value::Num(value))
                );
                for target in [
                    super::super::workspace::WorkspaceTarget::Current,
                    super::super::workspace::WorkspaceTarget::Caller,
                    super::super::workspace::WorkspaceTarget::Base,
                ] {
                    assert_eq!(
                        super::super::workspace::workspace_target_snapshot(target)
                            .unwrap()
                            .vars[0],
                        Value::Num(value)
                    );
                }
            })))
        }

        let base = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (first_runtime, first_service) = VmWorkspaceService::prepare(base.clone(), None);
        let (second_runtime, second_service) = VmWorkspaceService::prepare(base, None);
        assert_eq!(
            first_runtime.local_identity(),
            second_runtime.local_identity()
        );

        let mut first = sibling(first_runtime, first_service.clone(), 1.0);
        let mut second = sibling(second_runtime, second_service.clone(), 2.0);
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        assert!(poll(first.as_mut()).is_ready());
        let first_completion =
            super::super::workspace::take_workspace_completion(first_service.frame).unwrap();
        assert_eq!(first_completion.state.vars[0], Value::Num(1.0));
        assert_eq!(
            second_service.try_lookup("x").unwrap(),
            Some(Value::Num(2.0))
        );

        drop(second);
        let second_completion =
            super::super::workspace::take_workspace_completion(second_service.frame).unwrap();
        assert_eq!(second_completion.state.vars[0], Value::Num(2.0));
        assert!(first_service.try_lookup("x").is_err());
        assert!(second_service.try_lookup("x").is_err());
    }

    impl RuntimeWorkspaceService for CountingWorkspaceService {
        fn lookup(&self, name: &str) -> Option<Value> {
            self.inner.lookup(name)
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            self.inner.snapshot()
        }

        fn global_names(&self) -> Vec<String> {
            self.inner.global_names()
        }

        fn try_lookup(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
            self.inner.try_lookup(name)
        }

        fn try_snapshot(&self) -> Result<Vec<(String, Value)>, RuntimeError> {
            self.snapshots.set(self.snapshots.get() + 1);
            self.inner.try_snapshot()
        }

        fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
            self.inner.ensure_effect_authority()
        }

        fn ensure_mutation_authority(&self) -> Result<(), RuntimeError> {
            self.mutation_admissions
                .set(self.mutation_admissions.get() + 1);
            self.inner.ensure_mutation_authority()
        }

        fn workspace_revision(
            &self,
        ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, RuntimeError> {
            self.inner.workspace_revision()
        }

        fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
            self.inner.assign(name, value)
        }

        fn clear(&self) -> Result<(), RuntimeError> {
            self.inner.clear()
        }

        fn remove(&self, name: &str) -> Result<(), RuntimeError> {
            self.inner.remove(name)
        }
    }

    #[test]
    fn retained_service_targets_its_exact_frame_across_nested_execution() {
        let base = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (outer_runtime, outer_service) = VmWorkspaceService::prepare(base, None);

        futures::executor::block_on(workspace_scope(
            outer_runtime.clone(),
            &outer_service,
            async {
                let mut outer_values = vec![Value::Num(1.0)];
                let outer_guard = super::super::workspace::set_transient_workspace_state(
                    HashMap::from([("x".to_string(), 0)]),
                    HashSet::from(["x".to_string()]),
                    &mut outer_values,
                    FrameGlobalBindings::default(),
                );
                assert_eq!(
                    outer_service.try_lookup("x").unwrap(),
                    Some(Value::Num(1.0))
                );

                let (inner_runtime, inner_service) =
                    VmWorkspaceService::prepare(outer_runtime.clone(), None);
                workspace_scope(inner_runtime, &inner_service, async {
                    let mut inner_values = vec![Value::Num(2.0)];
                    let inner_guard = super::super::workspace::set_transient_workspace_state(
                        HashMap::from([("x".to_string(), 0)]),
                        HashSet::from(["x".to_string()]),
                        &mut inner_values,
                        FrameGlobalBindings::default(),
                    );
                    assert_eq!(
                        inner_service.try_lookup("x").unwrap(),
                        Some(Value::Num(2.0))
                    );
                    assert_eq!(
                        outer_service.try_lookup("x").unwrap(),
                        Some(Value::Num(1.0))
                    );
                    drop(inner_guard);
                })
                .await;

                let error = inner_service.try_lookup("x").unwrap_err();
                assert_eq!(error.identifier(), Some("RunMat:Workspace:FrameClosed"));
                assert_eq!(
                    outer_service.try_lookup("x").unwrap(),
                    Some(Value::Num(1.0))
                );
                drop(outer_guard);
            },
        ));

        let error = outer_service.try_lookup("x").unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Workspace:FrameClosed"));
    }

    #[test]
    fn unchanged_workspace_revision_does_not_rematerialize_the_frame() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base_service = runmat_runtime::workspace::BaseWorkspaceService::new(owner);
        base_service.assign("x", Value::Num(1.0)).unwrap();
        let inner: Rc<dyn RuntimeWorkspaceService> = Rc::new(base_service.clone());
        let delegate = Rc::new(CountingWorkspaceService {
            inner,
            snapshots: Cell::new(0),
            mutation_admissions: Cell::new(0),
        });
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(delegate.clone());
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);

        let call_runtime = runtime.clone();
        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(1.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(1.0)));
            assert_eq!(delegate.snapshots.get(), 1);
            assert_eq!(
                runmat_runtime::call_builtin_async_in_context(
                    &call_runtime,
                    "plus",
                    &[Value::Num(1.0), Value::Num(2.0)],
                )
                .await
                .unwrap(),
                Value::Num(3.0)
            );
            assert_eq!(
                delegate.snapshots.get(),
                1,
                "an unchanged post-call revision must not rematerialize the frame"
            );

            base_service.assign("x", Value::Num(3.0)).unwrap();
            runmat_runtime::call_builtin_async_in_context(
                &call_runtime,
                "plus",
                &[Value::Num(3.0), Value::Num(4.0)],
            )
            .await
            .unwrap();
            super::super::workspace::reconcile_workspace_vars(&mut values);
            assert_eq!(values[0], Value::Num(3.0));
            assert_eq!(service.lookup("x"), Some(Value::Num(3.0)));
            assert_eq!(delegate.snapshots.get(), 2);
            drop(guard);
        }));
    }

    #[test]
    fn exact_detach_rejects_synchronous_rebind_during_publication() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base = runmat_runtime::workspace::BaseWorkspaceService::new(owner);
        base.assign("x", Value::Num(1.0)).unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(Rc::new(base.clone()));
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);
        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(1.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            let detached = workspace_frame_detach_selected(
                service.frame,
                &WorkspaceDetachSelection::Names(vec!["x".into()]),
                |_| {
                    assert!(
                        workspace_frame_assign(service.frame, "x", Value::Num(2.0)).is_err(),
                        "publication cannot synchronously rebind the guarded frame"
                    );
                    assert!(
                        service.assign("x", Value::Num(2.0)).is_err(),
                        "service admission must fail before delegated publication"
                    );
                    assert_eq!(base.lookup("x"), Some(Value::Num(1.0)));
                    Ok(())
                },
            )
            .expect("exact transaction");
            assert_eq!(detached.entries(), &[("x".into(), Value::Num(1.0))]);
            service.assign("x", Value::Num(3.0)).unwrap();
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(3.0)));
            drop(guard);
        }));
    }

    #[test]
    fn nested_top_level_vm_detach_publishes_through_both_exact_frames() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base = runmat_runtime::workspace::BaseWorkspaceService::new(owner);
        base.assign("x", Value::Num(1.0)).unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(Rc::new(base.clone()));
        let (outer_runtime, outer) =
            VmWorkspaceService::prepare(runtime.with_service_ports(ports), None);
        futures::executor::block_on(workspace_scope(outer_runtime.clone(), &outer, async {
            let mut outer_values = vec![Value::Num(1.0)];
            let outer_guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut outer_values,
                FrameGlobalBindings::default(),
            );
            let (inner_runtime, inner) = VmWorkspaceService::prepare(outer_runtime, None);
            workspace_scope(inner_runtime, &inner, async {
                let mut inner_values = vec![Value::Num(1.0)];
                let inner_guard = super::super::workspace::set_workspace_state(
                    HashMap::from([("x".to_string(), 0)]),
                    HashSet::from(["x".to_string()]),
                    &mut inner_values,
                    FrameGlobalBindings::default(),
                );
                let detached = inner
                    .detach_selected(&WorkspaceDetachSelection::Names(vec!["x".into()]))
                    .expect("nested exact publication");
                assert_eq!(detached.entries(), &[("x".into(), Value::Num(1.0))]);
                assert_eq!(inner.lookup("x"), None);
                assert_eq!(outer.lookup("x"), None);
                assert_eq!(base.lookup("x"), None);
                drop(inner_guard);
            })
            .await;
            drop(outer_guard);
        }));
    }

    #[test]
    fn workspace_guard_drop_during_publication_defers_exact_frame_retirement() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);
        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(1.0)];
            let mut guard = Some(super::super::workspace::set_transient_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            ));
            let detached = workspace_frame_detach_selected(
                service.frame,
                &WorkspaceDetachSelection::All,
                |_| {
                    drop(guard.take());
                    assert_eq!(service.lookup("x"), Some(Value::Num(1.0)));
                    Ok(())
                },
            )
            .expect("frame stays published through exact detach commit");
            assert_eq!(detached.entries(), &[("x".into(), Value::Num(1.0))]);
            assert!(
                service.try_lookup("x").is_err(),
                "deferred close runs after commit"
            );
        }));
    }

    #[test]
    fn top_level_global_declaration_removes_shadowed_base_and_is_captured_typed() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base_service = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
        base_service.assign("g", Value::Num(1.0)).unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime.service_ports().clone().with_workspace(base_service);
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);

        let completed = futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(1.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("g".to_string(), 0)]),
                HashSet::from(["g".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            service.declare_global_binding(0, "g").unwrap();
            assert_eq!(
                service.try_lookup("g").unwrap(),
                Some(Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()
                )),
                "a first global declaration must not expose the shadowed local"
            );
            service
                .assign_executor_slot(0, "g", Value::Num(9.0))
                .unwrap();
            drop(guard);
            super::super::workspace::take_updated_workspace_state().unwrap()
        }));

        let projection = owner.snapshot().unwrap();
        assert_eq!(projection.get("g"), None);
        assert_eq!(
            completed.global_aliases.get(&0),
            Some(&runmat_runtime::workspace::bindings::FrameGlobalBinding::Named("g".into()))
        );
    }

    #[test]
    fn transient_frame_rejects_effects_after_session_workspace_closes() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base_service = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime.service_ports().clone().with_workspace(base_service);
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);

        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(1.0)];
            let guard = super::super::workspace::set_transient_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            owner.close().unwrap();

            let error = service.try_lookup("x").unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
            let error = service.assign("x", Value::Num(2.0)).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
            let error = service
                .assign_executor_slot(0, "x", Value::Num(3.0))
                .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
            let error = service.remove("x").unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
            let error = service.clear().unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
            assert_eq!(service.lookup("x"), Some(Value::Num(1.0)));
            drop(guard);
        }));
    }

    #[test]
    fn executor_slot_stores_do_not_snapshot_the_canonical_workspace() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let setup = runmat_runtime::workspace::BaseWorkspaceService::new(owner.clone());
        for index in 0..256 {
            setup
                .assign(&format!("seed_{index}"), Value::Num(index as f64))
                .unwrap();
        }
        let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
            owner.begin_projection().unwrap(),
        );
        let projection: Rc<dyn RuntimeWorkspaceService> = Rc::new(projection);
        let delegate = Rc::new(CountingWorkspaceService {
            inner: projection,
            snapshots: Cell::new(0),
            mutation_admissions: Cell::new(0),
        });
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(delegate.clone());
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);

        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let mut values = vec![Value::Num(0.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            for value in 1..=64 {
                service
                    .assign_executor_slot(0, "x", Value::Num(value as f64))
                    .unwrap();
            }
            assert_eq!(delegate.mutation_admissions.get(), 64);
            assert_eq!(delegate.snapshots.get(), 0);
            assert_eq!(service.lookup("x"), Some(Value::Num(64.0)));
            drop(guard);
        }));
    }

    #[test]
    fn canonical_reconciliation_never_overwrites_a_global_alias() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let base_service = runmat_runtime::workspace::BaseWorkspaceService::new(owner);
        base_service.assign("x", Value::Num(1.0)).unwrap();
        base_service.assign("unrelated", Value::Num(2.0)).unwrap();
        let inner: Rc<dyn RuntimeWorkspaceService> = Rc::new(base_service.clone());
        let delegate = Rc::new(CountingWorkspaceService {
            inner,
            snapshots: Cell::new(0),
            mutation_admissions: Cell::new(0),
        });
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(delegate.clone());
        let runtime = runtime.with_service_ports(ports);
        let (runtime, service) = VmWorkspaceService::prepare(runtime, None);

        futures::executor::block_on(workspace_scope(runtime, &service, async {
            let aliases = FrameGlobalBindings::default();
            aliases.declare_named(0, "x".into());
            runmat_runtime::workspace::session::store_global_named("x", Value::Num(9.0));
            let mut values = vec![Value::Num(9.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                aliases,
            );

            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(9.0)));
            assert_eq!(
                runmat_runtime::workspace::session::global_value("x"),
                Some(Value::Num(9.0))
            );
            base_service.assign("unrelated", Value::Num(3.0)).unwrap();
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(9.0)));
            assert_eq!(
                runmat_runtime::workspace::session::global_value("x"),
                Some(Value::Num(9.0)),
                "a base-workspace revision must not write through the global alias"
            );
            drop(guard);
        }));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn closed_frame_releases_its_canonical_delegate_while_service_is_retained() {
        runmat_gc::gc_test_context(|| {
            let allocation =
                runmat_gc::gc_allocate_rooted(Value::String("canonical".into())).unwrap();
            let handle = allocation.handle();
            let retained = Value::HandleObject(runmat_value::HandleRef {
                class_name: "VmWorkspaceGcFixture".into(),
                target: handle,
                valid: true,
            });
            let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
            let mut initial = runmat_runtime::workspace::WorkspaceDelta::new();
            initial.assign("canonical_only", retained).unwrap();
            owner.apply(initial).unwrap();
            let projection = Rc::new(
                runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
                    owner.begin_projection().unwrap(),
                ),
            );
            let canonical: Rc<dyn RuntimeWorkspaceService> = projection.clone();
            let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let ports = runtime
                .service_ports()
                .clone()
                .with_workspace(canonical.clone());
            let runtime = runtime.with_service_ports(ports);
            let (runtime, service) = VmWorkspaceService::prepare(runtime, None);
            drop(canonical);
            drop(projection);
            drop(owner);
            allocation.unroot().unwrap();

            futures::executor::block_on(workspace_scope(runtime, &service, async {
                let mut values = Vec::new();
                let guard = super::super::workspace::set_workspace_state(
                    HashMap::new(),
                    HashSet::new(),
                    &mut values,
                    FrameGlobalBindings::default(),
                );
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_ok());

                drop(guard);
                runmat_gc::gc_collect_major().unwrap();
                assert!(
                    runmat_gc::gc_clone_value(&handle).is_err(),
                    "a closed VM frame must release its canonical delegate"
                );
                assert_eq!(
                    service.try_snapshot().unwrap_err().identifier(),
                    Some("RunMat:Workspace:FrameClosed")
                );
            }));
        });
    }
}
