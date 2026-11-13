//! Explicit superclass calls retain the dynamic receiver and select a body
//! from its frozen owner graph, independently of virtual dispatch.
use super::*;
use std::rc::Rc;

pub(crate) async fn call_selected_super_method(
    runtime: Option<&crate::context::RuntimeContext>,
    caller: &ClassIdentity,
    superclass: &ClassIdentity,
    name: &runmat_types::MethodName,
    args: Vec<Value>,
    facts: Rc<runmat_value::SelectedObjectClassFacts>,
) -> Result<RootedValue, RuntimeError> {
    let runtime = runtime.ok_or_else(|| {
        semantic_error(
            "SelectedObjectMethodContextMissing",
            "superclass call has no issuing runtime",
        )
    })?;
    if !runtime.admits_selected_method_caller(caller, &facts) {
        return Err(semantic_error(
            "MethodOwnerInvalid",
            "superclass call differs from its active receiver method",
        ));
    }
    let method = facts
        .super_method(caller, superclass, name)
        .ok_or_else(|| {
            semantic_error(
                "UndefinedFunction",
                "selected superclass has no admitted method",
            )
        })?;
    if method.owner.is_static || method.is_abstract {
        return Err(semantic_error(
            "MethodStaticAccess",
            "superclass calls require a concrete instance method",
        ));
    }
    if method.access == runmat_types::MemberAccess::Private
        && caller != &method.owner.declaring_class
    {
        return Err(semantic_error(
            "MethodPrivate",
            "superclass method is private",
        ));
    }
    let binding = method.binding.clone().ok_or_else(|| {
        semantic_error(
            "ClassMethodBindingInvalid",
            "superclass method lacks its exact definition lease",
        )
    })?;
    let selected = runtime.clone().with_selected_super_method_body(
        Rc::clone(&facts),
        method.owner.clone(),
        binding.clone(),
    );
    let descriptor = CallableDescriptor::bound(binding, args, 1, CallableCallKind::Direct);
    crate::call::descriptor::execute_callable_descriptor_in_context(&selected, descriptor)
        .await?
        .try_into_single()
}
