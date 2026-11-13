use super::super::{save_error_with_source, save_workspace_flow, SAVE_ERROR_WORKSPACE};
use crate::{workspace, BuiltinResult};
use runmat_value::Value;

pub(in crate::builtins::io::mat::save) struct WorkspaceReadSet<'a> {
    target: WorkspaceReadTarget<'a>,
    _roots: Vec<crate::execution::RootedValueHandles>,
}

enum WorkspaceReadTarget<'a> {
    Service {
        service: std::rc::Rc<dyn crate::context::RuntimeWorkspaceService>,
        runtime: &'a crate::context::RuntimeContext,
    },
    Facade,
}

impl<'a> WorkspaceReadSet<'a> {
    pub(in crate::builtins::io::mat::save) fn capture(
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let target = match runtime.and_then(|runtime| {
            runtime
                .service_ports()
                .workspace()
                .map(|service| (runtime, service))
        }) {
            Some((runtime, service)) => {
                service
                    .ensure_effect_authority_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                WorkspaceReadTarget::Service {
                    service: service.clone(),
                    runtime,
                }
            }
            None => WorkspaceReadTarget::Facade,
        };
        Ok(Self {
            target,
            _roots: Vec::new(),
        })
    }

    pub(in crate::builtins::io::mat::save) fn liveness_service(
        &self,
    ) -> Option<std::rc::Rc<dyn crate::context::RuntimeWorkspaceService>> {
        match &self.target {
            WorkspaceReadTarget::Service { service, .. } => Some(service.clone()),
            WorkspaceReadTarget::Facade => None,
        }
    }

    pub(in crate::builtins::io::mat::save) fn snapshot(
        &self,
    ) -> BuiltinResult<Option<Vec<(String, Value)>>> {
        match &self.target {
            WorkspaceReadTarget::Service { service, runtime } => {
                service
                    .ensure_effect_authority_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                let snapshot = service
                    .try_snapshot_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                service
                    .ensure_effect_authority_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                Ok(Some(snapshot))
            }
            WorkspaceReadTarget::Facade => workspace::try_snapshot().map_err(save_workspace_flow),
        }
    }

    pub(in crate::builtins::io::mat::save) fn lookup(
        &self,
        name: &str,
    ) -> BuiltinResult<Option<Value>> {
        match &self.target {
            WorkspaceReadTarget::Service { service, runtime } => {
                service
                    .ensure_effect_authority_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                let value = service
                    .try_lookup_with_context(runtime, name)
                    .map_err(save_workspace_flow)?;
                service
                    .ensure_effect_authority_with_context(runtime)
                    .map_err(save_workspace_flow)?;
                Ok(value)
            }
            WorkspaceReadTarget::Facade => workspace::try_lookup(name).map_err(save_workspace_flow),
        }
    }

    pub(in crate::builtins::io::mat::save) fn retain_values<'value>(
        &mut self,
        values: impl Iterator<Item = &'value Value>,
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
