use runmat_value::Value;

use super::{
    base, resolver_storage, with_context_resolver, BaseWorkspaceOwner, BaseWorkspaceProjection,
    DetachedWorkspaceValues, WorkspaceDetachSelection,
};

/// Exact current-service retirement. The legacy resolver has no binding
/// identity, so it cannot authorize a callback-bearing clear operation.
pub fn try_detach_selected(
    selection: &WorkspaceDetachSelection,
) -> Result<DetachedWorkspaceValues, crate::RuntimeError> {
    let context = crate::context::legacy::active().ok_or_else(workspace_capability_error)?;
    let service = context
        .service_ports()
        .workspace()
        .ok_or_else(workspace_capability_error)?;
    service.detach_selected_with_context(&context, selection)
}

/// Checked lookup for code that must distinguish an empty workspace from an
/// unavailable or closed workspace authority. An installed service is the
/// complete authority: its error is returned directly and never falls through
/// to a legacy resolver.
pub fn try_lookup(name: &str) -> Result<Option<Value>, crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.try_lookup_with_context(&context, name);
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return Ok((resolver.lookup)(name));
        }
    }
    Ok(resolver_storage::with(|resolver| {
        resolver.and_then(|resolver| (resolver.lookup)(name))
    }))
}

/// Checked snapshot with `None` reserved for the absence of any workspace
/// resolver. Installed service failures remain typed and are not interpreted
/// as an empty workspace.
pub fn try_snapshot() -> Result<Option<Vec<(String, Value)>>, crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.try_snapshot_with_context(&context).map(Some);
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return Ok(Some((resolver.snapshot)()));
        }
    }
    Ok(resolver_storage::with(|resolver| {
        resolver.map(|resolver| (resolver.snapshot)())
    }))
}

/// Read session-global names associated with the active RuntimeContext. The
/// resolver-storage branch is a standalone compatibility path for execution
/// hosts not yet composed through RuntimeServicePorts; it is not canonical
/// session authority.
pub fn try_global_names() -> Result<Vec<String>, crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.try_global_names_with_context(&context);
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return Ok((resolver.globals)());
        }
    }
    Ok(resolver_storage::with(|resolver| {
        resolver
            .map(|resolver| (resolver.globals)())
            .unwrap_or_default()
    }))
}

/// Resolve the canonical session base-workspace owner. Procedure-local and
/// legacy resolver services do not implicitly acquire this authority.
pub fn base_workspace_owner() -> Result<BaseWorkspaceOwner, crate::RuntimeError> {
    let context = crate::context::legacy::active().ok_or_else(base_workspace_capability_error)?;
    let service = context
        .service_ports()
        .require_base_workspace("access the canonical base workspace")
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?;
    Ok(service.owner())
}

fn base_workspace_capability_error() -> crate::RuntimeError {
    crate::context::RuntimeCapabilityError::new(
        crate::context::RuntimeCapability::BaseWorkspace,
        "access the canonical base workspace",
    )
    .into_runtime_error()
}

pub fn begin_base_workspace_projection() -> Result<BaseWorkspaceProjection, crate::RuntimeError> {
    base_workspace_owner()?
        .begin_projection()
        .map_err(base::base_workspace_error_to_runtime)
}

pub fn try_assign(name: &str, value: Value) -> Result<(), crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.assign_with_context(&context, name, value);
        }
        if context.state().workspace.borrow().is_some() {
            return with_context_resolver(&context, |resolver| {
                let assign = resolver.assign.ok_or_else(|| {
                    operation_unavailable("workspace assignment is unavailable in this invocation")
                })?;
                assign(name, value)
            });
        }
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(workspace_capability_error)?;
        let assign = resolver.assign.ok_or_else(|| {
            operation_unavailable("workspace assignment is unavailable in this invocation")
        })?;
        assign(name, value)
    })
}

pub fn try_clear() -> Result<(), crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.clear_with_context(&context);
        }
        if context.state().workspace.borrow().is_some() {
            return with_context_resolver(&context, |resolver| {
                let clear = resolver.clear.ok_or_else(|| {
                    operation_unavailable("workspace clearing is unavailable in this invocation")
                })?;
                clear()
            });
        }
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(workspace_capability_error)?;
        let clear = resolver.clear.ok_or_else(|| {
            operation_unavailable("workspace clearing is unavailable in this invocation")
        })?;
        clear()
    })
}

pub fn try_remove(name: &str) -> Result<(), crate::RuntimeError> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.remove_with_context(&context, name);
        }
        if context.state().workspace.borrow().is_some() {
            return with_context_resolver(&context, |resolver| {
                let remove = resolver.remove.ok_or_else(|| {
                    operation_unavailable("workspace removal is unavailable in this invocation")
                })?;
                remove(name)
            });
        }
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(workspace_capability_error)?;
        let remove = resolver.remove.ok_or_else(|| {
            operation_unavailable("workspace removal is unavailable in this invocation")
        })?;
        remove(name)
    })
}

fn workspace_capability_error() -> crate::RuntimeError {
    crate::context::RuntimeCapabilityError::new(
        crate::context::RuntimeCapability::Workspace,
        "access or mutate the canonical workspace",
    )
    .into_runtime_error()
}

fn operation_unavailable(message: &'static str) -> crate::RuntimeError {
    crate::runtime_error::semantic_error("RunMat:Workspace:OperationUnavailable", message)
}
