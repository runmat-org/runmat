use std::cell::{Cell, RefCell};
use std::rc::Rc;

use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::workspace::{
    reconcile_workspace_frame, release_pending_workspace_service_frame,
    reserve_workspace_service_frame, workspace_frame_assign, workspace_frame_assign_from_executor,
    workspace_frame_clear, workspace_frame_lookup, workspace_frame_name_is_global,
    workspace_frame_publishes, workspace_frame_remove, workspace_frame_snapshot,
    workspace_global_names, WorkspaceFrameId, WorkspaceFrameLifecycle,
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
}

impl VmWorkspaceService {
    pub(crate) fn install(
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> runmat_runtime::context::RuntimeContext {
        Self::prepare(runtime).0
    }

    fn prepare(
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> (
        runmat_runtime::context::RuntimeContext,
        Rc<VmWorkspaceService>,
    ) {
        let delegate = runtime.service_ports().workspace().cloned();
        let runtime_id = runtime.local_identity();
        let pending = reserve_workspace_service_frame(runtime_id);
        *pending.delegate.borrow_mut() = delegate;
        let service = Rc::new(Self {
            delegate: pending.delegate,
            runtime: runtime_id,
            frame: pending.id,
            lifecycle: pending.lifecycle,
            delegate_revision: Cell::new(None),
        });
        let port: Rc<dyn RuntimeWorkspaceService> = service.clone();
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
            WorkspaceFrameLifecycle::Active => workspace_global_names(),
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
        self.delegate()
            .map(|delegate| delegate.try_global_names())
            .unwrap_or_else(|| Ok(workspace_global_names()))
    }

    fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.ensure_frame_active()?;
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
        if let Some(delegate) = self.delegate() {
            delegate.ensure_mutation_authority()?;
        }
        Ok(())
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()?;
        runmat_runtime::execution::validate_storable_value(&value)?;
        if self.should_publish()
            && !workspace_frame_name_is_global(self.frame, name).unwrap_or(false)
        {
            if let Some(delegate) = self.delegate() {
                delegate.assign(name, value.clone())?;
            }
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
        if self.should_publish()
            && !workspace_frame_name_is_global(self.frame, name).unwrap_or(false)
        {
            if let Some(delegate) = self.delegate() {
                delegate.assign(name, value.clone())?;
            }
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

    fn remove(&self, name: &str) -> Result<(), RuntimeError> {
        self.ensure_effect_authority()?;
        if self.should_publish()
            && !workspace_frame_name_is_global(self.frame, name).unwrap_or(false)
        {
            if let Some(delegate) = self.delegate() {
                delegate.remove(name)?;
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

    use super::*;

    struct CountingWorkspaceService {
        inner: Rc<dyn RuntimeWorkspaceService>,
        snapshots: Cell<usize>,
        mutation_admissions: Cell<usize>,
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
        let (outer_runtime, outer_service) = VmWorkspaceService::prepare(base);

        futures::executor::block_on(outer_runtime.scope(async {
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

            let (inner_runtime, inner_service) = VmWorkspaceService::prepare(outer_runtime.clone());
            inner_runtime
                .scope(async {
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
        }));

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
        let (runtime, service) = VmWorkspaceService::prepare(runtime);

        futures::executor::block_on(runtime.scope(async {
            let mut values = vec![Value::Num(1.0)];
            let guard = super::super::workspace::set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut values,
                FrameGlobalBindings::default(),
            );
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(1.0)));
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(1.0)));
            assert_eq!(delegate.snapshots.get(), 1);

            base_service.assign("x", Value::Num(3.0)).unwrap();
            assert_eq!(service.try_lookup("x").unwrap(), Some(Value::Num(3.0)));
            assert_eq!(delegate.snapshots.get(), 2);
            drop(guard);
        }));
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
        let (runtime, service) = VmWorkspaceService::prepare(runtime);

        futures::executor::block_on(runtime.scope(async {
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
        let (runtime, service) = VmWorkspaceService::prepare(runtime);

        futures::executor::block_on(runtime.scope(async {
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
        let (runtime, service) = VmWorkspaceService::prepare(runtime);

        futures::executor::block_on(runtime.scope(async {
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
            let (runtime, service) = VmWorkspaceService::prepare(runtime);
            drop(canonical);
            drop(projection);
            drop(owner);
            allocation.unroot().unwrap();

            futures::executor::block_on(runtime.scope(async {
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
