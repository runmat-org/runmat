//! Explicit `evalin('base', ...)` authority. A caller's VM frame is not the
//! canonical base workspace merely because it is the first frame on a stack.

use std::collections::{HashMap, HashSet};

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::bindings::FrameGlobalBinding;
use runmat_runtime::workspace::{
    base_workspace_error_to_runtime, BaseWorkspaceOwner, GlobalBindingTarget,
    WorkspaceCompletionIdentity,
};
use runmat_runtime::RuntimeError;
#[cfg(test)]
use runmat_value::Value;

use crate::runtime::workspace::{
    WorkspaceAssignedReport, WorkspaceTargetSnapshot, WorkspaceValueSnapshot,
};

#[path = "dynamic_base_target/completion.rs"]
mod completion;

/// The selected service grants the owner; the captured revision owns the
/// conflict boundary for a later transactional publication.
pub(super) struct ExplicitBaseTarget {
    owner: BaseWorkspaceOwner,
    identity: WorkspaceCompletionIdentity,
    visible_globals: HashMap<String, GlobalBindingTarget>,
}

impl ExplicitBaseTarget {
    #[cfg(test)]
    pub(super) fn assign(
        runtime: &RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), RuntimeError> {
        let owner = runtime
            .service_ports()
            .require_base_workspace("assign explicit base workspace value")
            .map_err(runmat_runtime::context::RuntimeCapabilityError::into_runtime_error)?
            .owner();
        runmat_runtime::workspace::assign_logical_base_value_with_context(
            &owner,
            &runtime.session_variable_authority(),
            runtime,
            name,
            value,
        )
    }

    pub(super) fn capture(
        runtime: &RuntimeContext,
    ) -> Result<(Self, WorkspaceTargetSnapshot), RuntimeError> {
        let owner = runtime
            .service_ports()
            .require_base_workspace("evaluate explicit base workspace source")
            .map_err(runmat_runtime::context::RuntimeCapabilityError::into_runtime_error)?
            .owner();
        let identity = owner
            .workspace_completion_identity()
            .map_err(base_workspace_error_to_runtime)?;
        let projection = owner
            .begin_projection()
            .map_err(base_workspace_error_to_runtime)?;
        let mut names = HashMap::new();
        let mut assigned = HashSet::new();
        let mut vars = Vec::new();
        for (name, value) in projection.iter() {
            let slot = vars.len();
            names.insert(name.to_owned(), slot);
            assigned.insert(name.to_owned());
            vars.push(value.clone());
        }
        let mut global_names = projection
            .global_names()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        global_names.sort();
        let mut global_aliases = HashMap::new();
        let mut visible_globals = HashMap::new();
        for name in global_names {
            let Some(target) = projection.global_target(&name).cloned() else {
                continue;
            };
            let value = match &target {
                GlobalBindingTarget::Named(bound) => {
                    runtime.session_global_value(bound.as_str())?
                }
                GlobalBindingTarget::LegacySlot(slot) => {
                    runtime.session_global_slot_value(*slot)?
                }
            };
            let Some(value) = value else {
                continue;
            };
            let slot = vars.len();
            names.insert(name.clone(), slot);
            assigned.insert(name.clone());
            vars.push(value);
            match &target {
                GlobalBindingTarget::Named(bound) => {
                    global_aliases
                        .insert(slot, FrameGlobalBinding::Named(bound.as_str().to_owned()));
                }
                GlobalBindingTarget::LegacySlot(bound) => {
                    global_aliases.insert(slot, FrameGlobalBinding::LegacySlot(*bound));
                }
            }
            visible_globals.insert(name, target);
        }
        Ok((
            Self {
                owner,
                identity,
                visible_globals,
            },
            WorkspaceTargetSnapshot {
                names,
                assigned,
                vars: crate::interpreter::vars::InterpreterVars::from_owned(vars),
                global_aliases,
            },
        ))
    }

    pub(super) fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
        self.owner
            .ensure_effect_authority()
            .map_err(base_workspace_error_to_runtime)
    }

    pub(super) fn commit(
        &self,
        runtime: &RuntimeContext,
        snapshot: WorkspaceValueSnapshot,
        report: WorkspaceAssignedReport,
    ) -> Result<(), RuntimeError> {
        let source = completion::BaseCompletionSource::prepare(
            runtime,
            &snapshot,
            &report,
            &self.visible_globals,
        )?;
        self.owner
            .apply_workspace_completion(runtime, self.identity, &source)
            .map_err(base_workspace_error_to_runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::workspace::{bindings::FrameGlobalBindings, WorkspaceDelta};
    use std::rc::Rc;

    #[test]
    fn explicit_base_reads_and_publishes_through_captured_owner_revision() {
        let owner = BaseWorkspaceOwner::new();
        let mut seed = WorkspaceDelta::new();
        seed.assign("base_answer", Value::Num(11.0)).unwrap();
        owner.apply(seed).unwrap();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_base_workspace(Rc::new(
                runmat_runtime::workspace::BaseWorkspaceService::new(owner.clone()),
            )),
        );
        let (target, mut snapshot) = ExplicitBaseTarget::capture(&runtime).unwrap();
        let slot = snapshot.names["base_answer"];
        assert_eq!(*snapshot.vars.get(slot).unwrap(), Value::Num(11.0));
        let frame = crate::runtime::workspace::set_workspace_state(
            snapshot.names,
            snapshot.assigned,
            &mut snapshot.vars,
            FrameGlobalBindings::from_snapshot(snapshot.global_aliases),
        );
        crate::runtime::workspace::workspace_assign("base_answer", Value::Num(12.0)).unwrap();
        drop(frame);
        let completed = crate::runtime::workspace::take_updated_workspace_state().unwrap();
        let report = crate::runtime::workspace::take_updated_workspace_assigned_report().unwrap();
        target.commit(&runtime, completed, report).unwrap();
        assert_eq!(
            owner.snapshot().unwrap().get("base_answer"),
            Some(&Value::Num(12.0))
        );

        ExplicitBaseTarget::assign(&runtime, "base_answer", Value::Num(13.0)).unwrap();
        assert_eq!(
            owner.snapshot().unwrap().get("base_answer"),
            Some(&Value::Num(13.0))
        );

        let mut bind = WorkspaceDelta::new();
        bind.bind_global("shared_answer");
        owner.apply(bind).unwrap();
        runtime
            .session_variable_authority()
            .declare_global_with_context(&runtime, "shared_answer")
            .unwrap();
        ExplicitBaseTarget::assign(&runtime, "shared_answer", Value::Num(14.0)).unwrap();
        assert_eq!(
            runtime.session_global_value("shared_answer").unwrap(),
            Some(Value::Num(14.0))
        );

        futures::executor::block_on(runtime.scope(async {
            runmat_runtime::workspace::session::store_global_slot(7, Value::Num(23.0))
                .expect("seed selected legacy slot");
        }));
        let mut legacy = WorkspaceDelta::new();
        legacy.bind_global_target("legacy_answer", GlobalBindingTarget::LegacySlot(7));
        owner.apply(legacy).unwrap();
        let (_, imported) = ExplicitBaseTarget::capture(&runtime).unwrap();
        let local_slot = imported.names["legacy_answer"];
        assert_ne!(local_slot, 7);
        assert_eq!(*imported.vars.get(local_slot).unwrap(), Value::Num(23.0));
        assert_eq!(
            imported.global_aliases.get(&local_slot),
            Some(&FrameGlobalBinding::LegacySlot(7))
        );
    }
}
