use super::super::{save_error_with_source, save_workspace_flow, SAVE_ERROR_WORKSPACE};
use crate::{workspace, BuiltinResult};
use runmat_value::Value;

pub(super) struct WorkspaceReadSet {
    target: WorkspaceReadTarget,
    _roots: Vec<crate::execution::RootedValueHandles>,
}

enum WorkspaceReadTarget {
    Service(std::rc::Rc<dyn crate::context::RuntimeWorkspaceService>),
    Facade,
}

impl WorkspaceReadSet {
    pub(super) fn capture(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<Self> {
        let target = match runtime.and_then(|runtime| runtime.service_ports().workspace()) {
            Some(service) => {
                service
                    .ensure_effect_authority()
                    .map_err(save_workspace_flow)?;
                WorkspaceReadTarget::Service(service.clone())
            }
            None => WorkspaceReadTarget::Facade,
        };
        Ok(Self {
            target,
            _roots: Vec::new(),
        })
    }

    pub(super) fn liveness_service(
        &self,
    ) -> Option<std::rc::Rc<dyn crate::context::RuntimeWorkspaceService>> {
        match &self.target {
            WorkspaceReadTarget::Service(service) => Some(service.clone()),
            WorkspaceReadTarget::Facade => None,
        }
    }

    pub(super) fn snapshot(&self) -> BuiltinResult<Option<Vec<(String, Value)>>> {
        match &self.target {
            WorkspaceReadTarget::Service(service) => {
                service
                    .ensure_effect_authority()
                    .map_err(save_workspace_flow)?;
                let snapshot = service.try_snapshot().map_err(save_workspace_flow)?;
                service
                    .ensure_effect_authority()
                    .map_err(save_workspace_flow)?;
                Ok(Some(snapshot))
            }
            WorkspaceReadTarget::Facade => workspace::try_snapshot().map_err(save_workspace_flow),
        }
    }

    pub(super) fn lookup(&self, name: &str) -> BuiltinResult<Option<Value>> {
        match &self.target {
            WorkspaceReadTarget::Service(service) => {
                service
                    .ensure_effect_authority()
                    .map_err(save_workspace_flow)?;
                let value = service.try_lookup(name).map_err(save_workspace_flow)?;
                service
                    .ensure_effect_authority()
                    .map_err(save_workspace_flow)?;
                Ok(value)
            }
            WorkspaceReadTarget::Facade => workspace::try_lookup(name).map_err(save_workspace_flow),
        }
    }

    pub(super) fn retain_values<'a>(
        &mut self,
        values: impl Iterator<Item = &'a Value>,
    ) -> BuiltinResult<()> {
        let roots = crate::execution::RootedValueHandles::for_values(values).map_err(|error| {
            save_error_with_source(
                &SAVE_ERROR_WORKSPACE,
                format!("save: could not retain workspace values: {error}"),
                error,
            )
        })?;
        self._roots.push(roots);
        Ok(())
    }
}
