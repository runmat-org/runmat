use runmat_value::Value;

use super::{base_workspace_error_to_runtime, BaseWorkspaceOwner, WorkspaceDelta};
use crate::workspace::session::SessionVariableAuthority;
use crate::RuntimeError;

/// Resolve a name in the logical base workspace without merging its two value
/// owners. A bound global reads from the exact session-variable authority;
/// every other name reads from canonical base storage.
pub fn lookup_logical_base_value(
    owner: &BaseWorkspaceOwner,
    session_variables: &SessionVariableAuthority,
    name: &str,
) -> Result<Option<Value>, RuntimeError> {
    if owner
        .is_global(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return session_variables.global_value(name);
    }
    owner.lookup(name).map_err(base_workspace_error_to_runtime)
}

/// Assign through the logical base workspace while preserving namespace
/// identity. Bound globals remain owned by the session-variable store; local
/// values are committed through the revisioned base-workspace owner.
pub fn assign_logical_base_value(
    owner: &BaseWorkspaceOwner,
    session_variables: &SessionVariableAuthority,
    name: &str,
    value: Value,
) -> Result<(), RuntimeError> {
    crate::execution::validate_storable_value(&value)?;
    if owner
        .is_global(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return session_variables.store_global(name, value);
    }
    let mut delta = WorkspaceDelta::new();
    delta
        .assign(name, value)
        .map_err(base_workspace_error_to_runtime)?;
    owner
        .apply(delta)
        .map(|_| ())
        .map_err(base_workspace_error_to_runtime)
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn logical_access_preserves_separate_local_and_global_value_owners() {
        let owner = BaseWorkspaceOwner::new();
        let runtime = runtime();
        let variables = runtime.session_variable_authority();

        assign_logical_base_value(&owner, &variables, "local", Value::Num(1.0)).unwrap();
        let mut binding = WorkspaceDelta::new();
        binding.bind_global("global");
        owner.apply(binding).unwrap();
        variables.store_global("global", Value::Num(2.0)).unwrap();

        assert_eq!(
            lookup_logical_base_value(&owner, &variables, "local").unwrap(),
            Some(Value::Num(1.0))
        );
        assert_eq!(
            lookup_logical_base_value(&owner, &variables, "global").unwrap(),
            Some(Value::Num(2.0))
        );
        assign_logical_base_value(&owner, &variables, "global", Value::Num(3.0)).unwrap();
        assert_eq!(
            variables.global_value("global").unwrap(),
            Some(Value::Num(3.0))
        );
        assert_eq!(owner.lookup("global").unwrap(), None);
    }

    #[test]
    fn logical_access_uses_the_explicit_session_authority() {
        let owner = BaseWorkspaceOwner::new();
        let mut binding = WorkspaceDelta::new();
        binding.bind_global("shared_name");
        owner.apply(binding).unwrap();
        let first_runtime = runtime();
        let second_runtime = runtime();
        let first = first_runtime.session_variable_authority();
        let second = second_runtime.session_variable_authority();
        first.store_global("shared_name", Value::Num(1.0)).unwrap();
        second.store_global("shared_name", Value::Num(2.0)).unwrap();

        assert_eq!(
            lookup_logical_base_value(&owner, &first, "shared_name").unwrap(),
            Some(Value::Num(1.0))
        );
        assert_eq!(
            lookup_logical_base_value(&owner, &second, "shared_name").unwrap(),
            Some(Value::Num(2.0))
        );
    }

    #[test]
    fn closed_base_authority_rejects_bound_global_access_before_side_effects() {
        let owner = BaseWorkspaceOwner::new();
        let runtime = runtime();
        let variables = runtime.session_variable_authority();
        let mut binding = WorkspaceDelta::new();
        binding.bind_global("global");
        owner.apply(binding).unwrap();
        variables.store_global("global", Value::Num(4.0)).unwrap();
        owner.close().unwrap();

        let read = lookup_logical_base_value(&owner, &variables, "global").unwrap_err();
        assert_eq!(read.identifier(), Some("RunMat:Workspace:Closed"));
        let write =
            assign_logical_base_value(&owner, &variables, "global", Value::Num(5.0)).unwrap_err();
        assert_eq!(write.identifier(), Some("RunMat:Workspace:Closed"));
        assert_eq!(
            variables.global_value("global").unwrap(),
            Some(Value::Num(4.0))
        );
    }
}
