use std::rc::Rc;

use runmat_value::Value;

use crate::context::{
    RuntimeCapability, RuntimeContext, RuntimeServicePorts, RuntimeWorkspaceService,
};
use crate::execution::RuntimeExecutionService;

use super::{
    BaseWorkspaceOwner, BaseWorkspaceProjectionService, BaseWorkspaceService, WorkspaceChangeKind,
    WorkspaceRevision,
};

#[derive(Debug, Default)]
struct CallerWorkspace;

impl RuntimeWorkspaceService for CallerWorkspace {
    fn lookup(&self, _name: &str) -> Option<Value> {
        None
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        Vec::new()
    }

    fn global_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn assign(&self, _name: &str, _value: Value) -> Result<(), crate::RuntimeError> {
        Ok(())
    }

    fn clear(&self) -> Result<(), crate::RuntimeError> {
        Ok(())
    }

    fn remove(&self, _name: &str) -> Result<(), crate::RuntimeError> {
        Ok(())
    }
}

fn runtime(services: RuntimeServicePorts) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(services)
}

#[test]
fn procedure_overlay_removal_retains_explicit_session_base_authority() {
    let owner = BaseWorkspaceOwner::new();
    let base = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let services = RuntimeServicePorts::default()
        .with_base_workspace(base)
        .with_workspace(Rc::new(CallerWorkspace));
    let procedure = runtime(services.without_workspace());
    assert!(procedure.service_ports().workspace().is_none());

    let _scope = procedure.enter();
    let admitted = crate::workspace::base_workspace_owner().expect("session base authority");
    let mut delta = super::WorkspaceDelta::new();
    delta.assign("from_procedure", Value::Num(1.0)).unwrap();
    admitted.apply(delta).unwrap();

    assert_eq!(owner.revision().unwrap().get(), 1);
    assert_eq!(
        owner.lookup("from_procedure").unwrap(),
        Some(Value::Num(1.0))
    );
}

#[test]
fn current_workspace_alone_cannot_manufacture_base_authority() {
    let current = runtime(RuntimeServicePorts::default().with_workspace(Rc::new(CallerWorkspace)));
    let _scope = current.enter();

    let error = crate::workspace::base_workspace_owner().unwrap_err();
    assert_eq!(
        error.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    assert!(error
        .message()
        .contains(RuntimeCapability::BaseWorkspace.as_str()));
}

#[test]
fn spmd_style_cloned_ports_cannot_inherit_parent_workspace_authorities() {
    let owner = BaseWorkspaceOwner::new();
    let base = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let parent = runtime(
        RuntimeServicePorts::default()
            .with_base_workspace(base.clone())
            .with_workspace(base),
    );

    let lab = parent
        .fork_parallel_lab(parent.service_ports().clone())
        .expect("idle session fork");
    assert!(lab.service_ports().workspace().is_none());
    assert!(lab.service_ports().base_workspace().is_none());
    {
        let _scope = lab.enter();
        assert!(crate::workspace::base_workspace_owner().is_err());
    }

    assert_eq!(owner.revision().unwrap(), WorkspaceRevision::INITIAL);
    assert!(!owner.is_closed().unwrap());
}

#[test]
fn current_base_service_admits_projection_without_owner_exposure() {
    let owner = BaseWorkspaceOwner::new();
    let current: Rc<dyn RuntimeWorkspaceService> =
        Rc::new(BaseWorkspaceService::new(owner.clone()));
    let mut projection = current
        .begin_workspace_projection()
        .unwrap()
        .expect("owner-backed current workspace projection");
    projection.assign("projected", Value::Num(2.0)).unwrap();
    projection.publish().unwrap();

    assert_eq!(owner.lookup("projected").unwrap(), Some(Value::Num(2.0)));
}

#[test]
fn projection_service_stages_ordered_changes_until_a_safepoint() {
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap());

    service.assign("discarded", Value::Num(1.0)).unwrap();
    service.clear().unwrap();
    service.assign("kept", Value::Num(2.0)).unwrap();

    assert!(owner.snapshot().unwrap().is_empty());
    assert!(service.has_pending_changes().unwrap());
    let receipt = service.publish().unwrap();
    assert_eq!(receipt.change_count(), 2);
    assert_eq!(owner.lookup("discarded").unwrap(), None);
    assert_eq!(owner.lookup("kept").unwrap(), Some(Value::Num(2.0)));
}

#[test]
fn projection_service_coalesces_effects_across_publication_safepoints() {
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap());

    service.assign("x", Value::Num(1.0)).unwrap();
    service.ensure_effect_authority().unwrap();
    service.assign("y", Value::Num(2.0)).unwrap();
    service.publish().unwrap();

    let effect = service.take_published_changes().unwrap();
    assert_eq!(effect.len(), 2);
    assert_eq!(effect.changes()[0].name(), Some("x"));
    assert_eq!(effect.changes()[0].value(), Some(&Value::Num(1.0)));
    assert_eq!(effect.changes()[1].name(), Some("y"));
    assert_eq!(effect.changes()[1].value(), Some(&Value::Num(2.0)));
    assert!(service.take_published_changes().unwrap().is_empty());

    service.assign("discarded", Value::Num(3.0)).unwrap();
    service.publish().unwrap();
    service.clear().unwrap();
    service.assign("kept", Value::Num(4.0)).unwrap();
    service.publish().unwrap();
    let effect = service.take_published_changes().unwrap();
    assert_eq!(effect.len(), 2);
    assert_eq!(effect.changes()[0].kind(), WorkspaceChangeKind::Clear);
    assert_eq!(effect.changes()[1].name(), Some("kept"));
    assert_eq!(effect.changes()[1].value(), Some(&Value::Num(4.0)));
}

#[test]
fn projection_service_preserves_typed_close_failures_before_mutation() {
    let owner = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap());
    owner.close().unwrap();

    for error in [
        service.assign("late", Value::Num(1.0)).unwrap_err(),
        service.remove("late").unwrap_err(),
        service.clear().unwrap_err(),
        service.try_lookup("late").unwrap_err(),
    ] {
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }
    assert!(!service.has_pending_changes().unwrap());
}

#[test]
fn projection_service_clones_share_one_exact_invocation_view() {
    let owner = BaseWorkspaceOwner::new();
    let writer = BaseWorkspaceProjectionService::new(owner.begin_projection().unwrap());
    let reader = writer.clone();

    writer.assign("shared", Value::Num(3.0)).unwrap();
    assert_eq!(reader.try_lookup("shared").unwrap(), Some(Value::Num(3.0)));
    reader.publish().unwrap();
    assert_eq!(owner.lookup("shared").unwrap(), Some(Value::Num(3.0)));
}
