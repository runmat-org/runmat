//! Stable, borrowed logical workspace view for executor handoff.
//!
//! The Base and Session owners retain their independent physical claims and
//! managed-handle roots. This type neither clones language Values nor builds a
//! second combined name/value map.

use runmat_value::Value;

use super::session::{GlobalRevision, SessionGlobalGeneration, SessionVariableAuthority};
use super::{
    base_workspace_error_to_runtime, BaseWorkspaceOwner, BaseWorkspaceSnapshot,
    GlobalBindingTarget, WorkspaceRevision,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;

#[derive(Clone, Debug)]
pub struct LogicalWorkspaceGeneration {
    base: BaseWorkspaceSnapshot,
    session: SessionGlobalGeneration,
    global_revision: GlobalRevision,
}

impl LogicalWorkspaceGeneration {
    /// Fork an independent Base owner over the same immutable managed
    /// generation. Subsequent writes detach through the ordinary projection
    /// path; no Value/name/target payload is replayed into a second map.
    pub fn fork_base_owner(&self) -> Result<BaseWorkspaceOwner, RuntimeError> {
        self.base
            .fork_managed_owner()
            .map_err(base_workspace_error_to_runtime)
    }

    pub(crate) fn from_base_snapshot(
        context: &RuntimeContext,
        base: BaseWorkspaceSnapshot,
        session: &SessionVariableAuthority,
    ) -> Result<Self, RuntimeError> {
        let globals = session.capture_global_generation(context)?;
        let global_revision = globals.revision();
        Ok(Self {
            base,
            session: globals,
            global_revision,
        })
    }

    pub fn capture(
        context: &RuntimeContext,
        base: &BaseWorkspaceOwner,
        session: &SessionVariableAuthority,
    ) -> Result<Self, RuntimeError> {
        let base = base
            .capture_stable_with_context(context)
            .map_err(base_workspace_error_to_runtime)?;
        Self::from_base_snapshot(context, base, session)
    }

    pub fn base_revision(&self) -> WorkspaceRevision {
        self.base.revision()
    }

    pub fn global_revision(&self) -> GlobalRevision {
        self.global_revision
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.base.global_target(name)
    }

    /// Exact workspace spellings and targets, including a global whose
    /// current session value is absent.
    pub fn global_bindings(&self) -> impl Iterator<Item = (&str, &GlobalBindingTarget)> {
        self.base
            .global_names()
            .filter_map(|name| self.base.global_target(name).map(|target| (name, target)))
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        match self.base.global_target(name) {
            Some(GlobalBindingTarget::Named(target)) => self.session.named(target.as_str()),
            Some(GlobalBindingTarget::LegacySlot(slot)) => self.session.slot(*slot),
            None => self.base.get(name),
        }
    }

    pub fn read_lease(&self, name: &str) -> Option<super::WorkspaceValueReadLease> {
        match self.base.global_target(name) {
            Some(GlobalBindingTarget::Named(target)) => self.session.named_lease(target.as_str()),
            Some(GlobalBindingTarget::LegacySlot(slot)) => self.session.slot_lease(*slot),
            None => self.base.read_lease(name),
        }
    }

    /// Sorted, unique workspace spellings with their logical current values.
    /// A global whose exact session target is absent has no visible value.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        let mut locals = self.base.iter().peekable();
        let mut globals = self.base.global_names().peekable();
        std::iter::from_fn(move || loop {
            let local_name = locals.peek().map(|(name, _)| *name);
            let global_name = globals.peek().copied();
            let name = match (local_name, global_name) {
                (Some(local), Some(global)) if local <= global => local,
                (Some(_), Some(global)) => global,
                (Some(local), None) => local,
                (None, Some(global)) => global,
                (None, None) => return None,
            };
            if local_name == Some(name) {
                locals.next();
            }
            if global_name == Some(name) {
                globals.next();
            }
            if let Some(value) = self.get(name) {
                return Some((name, value));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use crate::workspace::{WorkspaceDelta, WorkspaceName};
    use std::rc::Rc;

    #[test]
    fn capture_retains_exact_named_and_slot_targets_across_later_writes() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let base = BaseWorkspaceOwner::new();
        let session = context.session_variable_authority();
        let tensor = runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let payload = tensor.as_f64_slice().unwrap().as_ptr();
        session
            .store_global_with_context(&context, "target", Value::Tensor(tensor))
            .unwrap();
        session
            .store_global_slot_with_context(&context, 7, Value::Num(7.0))
            .unwrap();
        let mut delta = WorkspaceDelta::new();
        delta.assign("local", Value::Num(1.0)).unwrap();
        delta.bind_global_target(
            "alias",
            GlobalBindingTarget::Named(WorkspaceName::unbounded("target")),
        );
        delta.bind_global_target("slot", GlobalBindingTarget::LegacySlot(7));
        base.apply(delta).unwrap();

        let captured = LogicalWorkspaceGeneration::capture(&context, &base, &session).unwrap();
        let captured_revision = captured.global_revision();
        let Value::Tensor(captured_tensor) = captured.get("alias").unwrap() else {
            panic!("named alias lost its target class");
        };
        assert_eq!(captured_tensor.as_f64_slice().unwrap().as_ptr(), payload);
        assert_eq!(captured.get("slot"), Some(&Value::Num(7.0)));
        assert_eq!(
            captured.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["alias", "local", "slot"]
        );

        session
            .store_global_with_context(&context, "target", Value::Num(9.0))
            .unwrap();
        session
            .store_global_slot_with_context(&context, 7, Value::Num(8.0))
            .unwrap();
        base.apply({
            let mut next = WorkspaceDelta::new();
            next.assign("local", Value::Num(2.0)).unwrap();
            next
        })
        .unwrap();
        assert_ne!(session.global_revision().unwrap(), captured_revision);
        assert_eq!(captured.get("local"), Some(&Value::Num(1.0)));
        assert_eq!(captured.get("slot"), Some(&Value::Num(7.0)));
        let Value::Tensor(retained) = captured.get("alias").unwrap() else {
            panic!("later session write changed a captured alias");
        };
        assert_eq!(retained.as_f64_slice().unwrap().as_ptr(), payload);
    }
}
