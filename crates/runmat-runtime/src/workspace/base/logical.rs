use runmat_value::Value;

use super::{base_workspace_error_to_runtime, BaseWorkspaceOwner, WorkspaceDelta};
use crate::workspace::session::SessionVariableAuthority;
use crate::workspace::GlobalBindingTarget;
use crate::RuntimeError;

/// Resolve a name in the logical base workspace without merging its two value
/// owners. A bound global reads from the exact session-variable authority;
/// every other name reads from canonical base storage.
pub fn lookup_logical_base_value(
    owner: &BaseWorkspaceOwner,
    session_variables: &SessionVariableAuthority,
    name: &str,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(target) = owner
        .global_target(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return match target {
            GlobalBindingTarget::Named(target_name) => {
                session_variables.global_value(target_name.as_str())
            }
            GlobalBindingTarget::LegacySlot(slot) => session_variables.global_slot(slot),
        };
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
    if let Some(target) = owner
        .global_target(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return match target {
            GlobalBindingTarget::Named(target_name) => {
                session_variables.store_global(target_name.as_str(), value)
            }
            GlobalBindingTarget::LegacySlot(slot) => {
                session_variables.store_global_slot(slot, value)
            }
        };
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

/// Context-bearing logical base assignment used by production callback
/// boundaries. Local values enter through the same admitted projection owner
/// as executor writes; bound globals remain in the session-variable owner.
pub fn assign_logical_base_value_with_context(
    owner: &BaseWorkspaceOwner,
    session_variables: &SessionVariableAuthority,
    context: &crate::context::RuntimeContext,
    name: &str,
    value: Value,
) -> Result<(), RuntimeError> {
    crate::execution::validate_storable_value(&value)?;
    if let Some(target) = owner
        .global_target(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return match target {
            GlobalBindingTarget::Named(target_name) => {
                session_variables.store_global_with_context(context, target_name.as_str(), value)
            }
            GlobalBindingTarget::LegacySlot(slot) => {
                session_variables.store_global_slot_with_context(context, slot, value)
            }
        };
    }
    let service = super::BaseWorkspaceProjectionService::new_for_context(
        owner
            .begin_projection()
            .map_err(base_workspace_error_to_runtime)?,
        context,
    );
    crate::context::RuntimeWorkspaceService::assign_with_context(&service, context, name, value)?;
    service.publish_with_context(context).map(|_| ())
}

/// Borrowed counterpart used by call boundaries whose arguments remain owned
/// by the caller. Every finite owner is prepared before any Value alias is
/// formed; explicit unbounded execution retains the legacy clone policy.
pub fn assign_logical_base_value_ref_with_context(
    owner: &BaseWorkspaceOwner,
    session_variables: &SessionVariableAuthority,
    context: &crate::context::RuntimeContext,
    name: &str,
    value: &Value,
) -> Result<(), RuntimeError> {
    crate::execution::validate_storable_value(value)?;
    if let Some(target) = owner
        .global_target(name)
        .map_err(base_workspace_error_to_runtime)?
    {
        return match target {
            GlobalBindingTarget::Named(target_name) => session_variables
                .store_global_ref_with_context(context, target_name.as_str(), value),
            GlobalBindingTarget::LegacySlot(slot) => {
                session_variables.store_global_slot_ref_with_context(context, slot, value)
            }
        };
    }
    let service = super::BaseWorkspaceProjectionService::new_for_context(
        owner
            .begin_projection()
            .map_err(base_workspace_error_to_runtime)?,
        context,
    );
    crate::context::RuntimeWorkspaceService::assign_ref_with_context(
        &service, context, name, value,
    )?;
    service.publish_with_context(context).map(|_| ())
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
        variables
            .store_global_with_context(&runtime, "global", Value::Num(2.0))
            .unwrap();

        assert_eq!(
            lookup_logical_base_value(&owner, &variables, "local").unwrap(),
            Some(Value::Num(1.0))
        );
        assert_eq!(
            lookup_logical_base_value(&owner, &variables, "global").unwrap(),
            Some(Value::Num(2.0))
        );
        assign_logical_base_value_with_context(
            &owner,
            &variables,
            &runtime,
            "global",
            Value::Num(3.0),
        )
        .unwrap();
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
        first
            .store_global_with_context(&first_runtime, "shared_name", Value::Num(1.0))
            .unwrap();
        second
            .store_global_with_context(&second_runtime, "shared_name", Value::Num(2.0))
            .unwrap();

        // A callback may run while another runtime is active. Its explicit
        // authority must determine both the destination and admission owner.
        futures::executor::block_on(second_runtime.scope(async {
            assign_logical_base_value_with_context(
                &owner,
                &first,
                &first_runtime,
                "shared_name",
                Value::Num(3.0),
            )
            .unwrap();
        }));

        assert_eq!(
            lookup_logical_base_value(&owner, &first, "shared_name").unwrap(),
            Some(Value::Num(3.0))
        );
        assert_eq!(
            lookup_logical_base_value(&owner, &second, "shared_name").unwrap(),
            Some(Value::Num(2.0))
        );
    }

    #[test]
    fn retargeted_name_and_slot_bindings_use_the_exact_session_destination() {
        let owner = BaseWorkspaceOwner::new();
        let runtime = runtime();
        let variables = runtime.session_variable_authority();
        variables
            .store_global_with_context(&runtime, "h", Value::Num(1.0))
            .unwrap();
        variables
            .store_global_with_context(&runtime, "j", Value::Num(2.0))
            .unwrap();
        variables
            .store_global_with_context(&runtime, "var_4", Value::Num(9.0))
            .unwrap();
        variables
            .store_global_slot_with_context(&runtime, 4, Value::Num(3.0))
            .unwrap();

        for (target, expected) in [
            (
                GlobalBindingTarget::Named(crate::workspace::WorkspaceName::unbounded("h")),
                1.0,
            ),
            (
                GlobalBindingTarget::Named(crate::workspace::WorkspaceName::unbounded("j")),
                2.0,
            ),
            (GlobalBindingTarget::LegacySlot(4), 3.0),
        ] {
            let mut delta = WorkspaceDelta::new();
            delta.bind_global_target("g", target.clone());
            owner.apply(delta).unwrap();
            assert_eq!(owner.global_target("g").unwrap(), Some(target));
            assert_eq!(
                lookup_logical_base_value(&owner, &variables, "g").unwrap(),
                Some(Value::Num(expected))
            );
        }
        assign_logical_base_value_with_context(&owner, &variables, &runtime, "g", Value::Num(4.0))
            .unwrap();
        assert_eq!(variables.global_slot(4).unwrap(), Some(Value::Num(4.0)));
        assert_eq!(
            variables.global_value("var_4").unwrap(),
            Some(Value::Num(9.0))
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
        variables
            .store_global_with_context(&runtime, "global", Value::Num(4.0))
            .unwrap();
        owner.close().unwrap();

        let read = lookup_logical_base_value(&owner, &variables, "global").unwrap_err();
        assert_eq!(read.identifier(), Some("RunMat:Workspace:Closed"));
        let write = assign_logical_base_value_with_context(
            &owner,
            &variables,
            &runtime,
            "global",
            Value::Num(5.0),
        )
        .unwrap_err();
        assert_eq!(write.identifier(), Some("RunMat:Workspace:Closed"));
        assert_eq!(
            variables.global_value("global").unwrap(),
            Some(Value::Num(4.0))
        );
    }
}
