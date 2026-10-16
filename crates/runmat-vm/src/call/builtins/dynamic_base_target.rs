//! Explicit `evalin('base', ...)` authority. A caller's VM frame is not the
//! canonical base workspace merely because it is the first frame on a stack.

use std::collections::{HashMap, HashSet};

use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::bindings::FrameGlobalBinding;
use runmat_runtime::workspace::{
    base_workspace_error_to_runtime, BaseWorkspaceOwner, WorkspaceDelta, WorkspaceRevision,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use crate::runtime::workspace::{
    active_named_global_bindings, WorkspaceAssignedReport, WorkspaceTargetSnapshot,
};

/// The selected service grants the owner; the captured revision owns the
/// conflict boundary for a later transactional publication.
pub(super) struct ExplicitBaseTarget {
    owner: BaseWorkspaceOwner,
    revision: WorkspaceRevision,
    visible_globals: HashSet<String>,
}

impl ExplicitBaseTarget {
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
        runmat_runtime::execution::validate_storable_value(&value)?;
        if owner
            .is_global(name)
            .map_err(base_workspace_error_to_runtime)?
        {
            owner
                .ensure_effect_authority()
                .map_err(base_workspace_error_to_runtime)?;
            runtime
                .session_variable_authority()
                .store_global(name, value)
        } else {
            let mut delta = WorkspaceDelta::new();
            delta
                .assign(name, value)
                .map_err(base_workspace_error_to_runtime)?;
            owner
                .apply(delta)
                .map(|_| ())
                .map_err(base_workspace_error_to_runtime)
        }
    }

    pub(super) fn capture(
        runtime: &RuntimeContext,
    ) -> Result<(Self, WorkspaceTargetSnapshot), RuntimeError> {
        let owner = runtime
            .service_ports()
            .require_base_workspace("evaluate explicit base workspace source")
            .map_err(runmat_runtime::context::RuntimeCapabilityError::into_runtime_error)?
            .owner();
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
        let mut visible_globals = HashSet::new();
        for name in global_names {
            let Some(value) = runtime.session_global_value(&name)? else {
                continue;
            };
            let slot = vars.len();
            names.insert(name.clone(), slot);
            assigned.insert(name.clone());
            vars.push(value);
            global_aliases.insert(slot, FrameGlobalBinding::Named(name.clone()));
            visible_globals.insert(name);
        }
        Ok((
            Self {
                owner,
                revision: projection.revision(),
                visible_globals,
            },
            WorkspaceTargetSnapshot {
                names,
                assigned,
                vars,
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
        vars: Vec<Value>,
        names: HashMap<String, usize>,
        assigned: HashSet<String>,
        global_aliases: HashMap<usize, FrameGlobalBinding>,
        report: WorkspaceAssignedReport,
    ) -> Result<(), RuntimeError> {
        let next_globals = active_named_global_bindings(&names, &global_aliases);
        let mut changed = report.names;
        changed.extend(report.removed_names);
        changed.extend(
            self.visible_globals
                .symmetric_difference(&next_globals)
                .cloned(),
        );
        let mut changed = changed.into_iter().collect::<Vec<_>>();
        changed.sort();
        if changed.is_empty() {
            return self.ensure_effect_authority();
        }
        let mut delta = WorkspaceDelta::new();
        for name in changed {
            if next_globals.contains(&name) {
                delta.bind_global(name);
            } else if assigned.contains(&name) {
                let slot = names.get(&name).copied().ok_or_else(|| {
                    runmat_runtime::runtime_error::semantic_error(
                        "RunMat:Workspace:TargetState",
                        "assigned base-workspace name has no execution slot",
                    )
                })?;
                let value = vars.get(slot).cloned().ok_or_else(|| {
                    runmat_runtime::runtime_error::semantic_error(
                        "RunMat:Workspace:TargetState",
                        "assigned base-workspace slot has no value",
                    )
                })?;
                runmat_runtime::execution::validate_storable_value(&value)?;
                delta
                    .assign(name, value)
                    .map_err(base_workspace_error_to_runtime)?;
            } else if self.visible_globals.contains(&name) {
                delta.unbind_global(name);
            } else {
                delta.remove(name);
            }
        }
        self.owner
            .commit(self.revision, delta)
            .map(|_| ())
            .map_err(base_workspace_error_to_runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(snapshot.vars[slot], Value::Num(11.0));
        snapshot.vars[slot] = Value::Num(12.0);
        let mut report = WorkspaceAssignedReport::default();
        report.names.insert("base_answer".to_owned());
        target
            .commit(
                snapshot.vars,
                snapshot.names,
                snapshot.assigned,
                snapshot.global_aliases,
                report,
            )
            .unwrap();
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
            .declare_global("shared_answer")
            .unwrap();
        ExplicitBaseTarget::assign(&runtime, "shared_answer", Value::Num(14.0)).unwrap();
        assert_eq!(
            runtime.session_global_value("shared_answer").unwrap(),
            Some(Value::Num(14.0))
        );
    }
}
