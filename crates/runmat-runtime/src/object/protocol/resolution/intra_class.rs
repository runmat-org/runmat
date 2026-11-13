use super::{semantic_error, ClassIdentity, ObjectAccessContext, ObjectProtocol, RuntimeError};
use runmat_value::SelectedObjectClassFacts;

/// Own-class indexing uses built-in access, but class visibility alone is not
/// evidence that this invocation is executing an actual method.
pub(super) fn uses_builtin_indexing(
    receiver: &ClassIdentity,
    protocol: ObjectProtocol,
    access: &ObjectAccessContext,
    selected: Option<&SelectedObjectClassFacts>,
) -> Result<bool, RuntimeError> {
    if !matches!(protocol, ObjectProtocol::Subsref | ObjectProtocol::Subsasgn) {
        return Ok(false);
    }
    if let Some(owner) = &access.selected_active_owner {
        if &owner.declaring_class != receiver
            && selected.is_none_or(|facts| !facts.ancestry().contains(&owner.declaring_class))
        {
            return Ok(false);
        }
        if access.caller_class.as_ref() != Some(&owner.declaring_class)
            || access.caller_method.as_ref() != Some(&owner.method)
            || !selected.is_some_and(|facts| {
                facts.class() == receiver
                    && match access.selected_method_dispatch {
                        crate::context::SelectedMethodDispatch::Virtual => {
                            ObjectAccessContext::selected_method_frame_is_admitted(owner, facts)
                                && selected_binding(facts, owner.role, &owner.method)
                                    == access.selected_active_binding.as_ref()
                        }
                        crate::context::SelectedMethodDispatch::Superclass => {
                            facts.method_by_owner(owner).is_some_and(|method| {
                                !owner.is_static
                                    && !method.is_abstract
                                    && method.binding.as_ref()
                                        == access.selected_active_binding.as_ref()
                            })
                        }
                    }
                    && access.selected_active_binding.is_some()
            })
        {
            return Err(semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                "active selected method differs from its receiver's class product",
            ));
        }
        return Ok(true);
    }
    if access.caller_class.as_ref() != Some(receiver) {
        return Ok(false);
    }
    let Some(active) = &access.active_method else {
        return Ok(false);
    };
    let current = crate::class_registry::lookup_bound_method(receiver, &active.method);
    if &active.declaring_class != receiver
        || access.caller_method.as_ref() != Some(&active.method)
        || !current.is_some_and(|bound| {
            bound.declaring_class == active.declaring_class
                && bound.registry_generation == active.registry_generation
                && bound.callable == active.callable
                && selected.is_none_or(|facts| {
                    selected_binding(facts, bound.declaration.role, &active.method)
                        .is_some_and(|binding| Some(binding) == bound.bound_function.as_ref())
                })
        })
    {
        return Err(semantic_error(
            "StaleObjectMethodFrame",
            "the active class method binding changed; rebuild or reload the class",
        ));
    }
    Ok(true)
}

fn selected_binding<'a>(
    facts: &'a SelectedObjectClassFacts,
    role: runmat_types::ClassMethodRole,
    method: &runmat_types::MethodName,
) -> Option<&'a runmat_value::BoundFunctionBinding> {
    match role {
        runmat_types::ClassMethodRole::Constructor => Some(facts.constructor()),
        runmat_types::ClassMethodRole::Ordinary => facts.method(method)?.binding.as_ref(),
    }
}
