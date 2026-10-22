use runmat_value::{ExactArrayIntoIter, ExactArrayStorage, Value};

use super::{resolve_single, SubscriptReadRequest};
use crate::execution::RootedValueSequence;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscript,
    ObjectSubscriptOrigin,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(super) async fn try_execute(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    steps: &mut ExactArrayIntoIter<ObjectSubscript>,
    request: &SubscriptReadRequest,
    caller_function_name: Option<&str>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    if !is_prefix(steps) {
        return Ok(None);
    }
    let member_step = steps.next().ok_or_else(|| {
        semantic_error(
            "InvalidObjectSubscriptPath",
            "dotted invocation has no member",
        )
    })?;
    let (ObjectIndexSelector::Member(member), semantic_fallback) = member_step.into_parts() else {
        unreachable!("validated dotted invocation has a member selector")
    };
    let paren_step = steps.next().ok_or_else(|| {
        semantic_error(
            "InvalidObjectSubscriptPath",
            "dotted invocation has no arguments",
        )
    })?;
    let (ObjectIndexSelector::IndexValues { components }, _) = paren_step.into_parts() else {
        unreachable!("validated dotted invocation has index-value arguments")
    };
    let requested_outputs = if steps.len() == 0 {
        request.known_requested_outputs()?.ok_or_else(|| {
            semantic_error(
                "SequenceOutputContextUnavailable",
                "dotted invocation requires an explicit output count",
            )
        })?
    } else {
        1
    };
    let result = if let Some(method) = crate::object::protocol::resolve_declared_object_method(
        &base,
        &runmat_types::MethodName::from(member.as_str()),
        &request.access,
    )? {
        let arguments: crate::call::BuiltinArguments = call_arguments(runtime, components)?.into();
        let arguments = arguments.try_insert(
            0,
            crate::execution::RootedValue::in_optional_context(base, runtime)?,
            runtime,
        )?;
        crate::object::protocol::invoke_resolved_object_method_retained(
            runtime,
            &method,
            arguments,
            requested_outputs,
        )
        .await?
    } else if crate::object::resolve::has_stored_member(&base, &member.as_str().into())?
        == crate::object::resolve::MemberPresence::Absent
        && (matches!(
            &base,
            Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) | Value::ClassRef(_)
        ) || semantic_fallback.is_some())
    {
        if crate::object::protocol::selected_receiver_has_authority(&base)? {
            return Err(semantic_error(
                "UndefinedFunction",
                "selected class has no admitted method for this member",
            ));
        }
        // This is a structural decision, not an exception-driven retry: a
        // missing member may use the established method-style callable route.
        let arguments = call_arguments(runtime, components)?;
        crate::object::dispatch::call_method_or_member_index_retained_with_context(
            runtime,
            base,
            member,
            arguments.into(),
            requested_outputs,
            caller_function_name,
            semantic_fallback,
        )
        .await?
    } else {
        let member_sequence = crate::object::resolve::read_member_sequence_text_with_context(
            runtime,
            base,
            member,
            false,
            caller_function_name,
        )
        .await?;
        let member_value = resolve_single(member_sequence)?;
        if super::callable::is_callable_value(&member_value) {
            let arguments = call_arguments(runtime, components)?;
            super::callable::invoke(
                runtime,
                member_value.value().clone(),
                arguments.into(),
                requested_outputs,
            )
            .await?
        } else {
            let step =
                ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues { components });
            super::read_default_step(
                runtime,
                member_value.value().clone(),
                &step,
                caller_function_name,
            )
            .await?
            .try_into_sequence()?
        }
    };
    Ok(Some(result))
}

fn is_prefix(steps: &ExactArrayIntoIter<ObjectSubscript>) -> bool {
    let (Some(member_step), Some(paren_step)) = (steps.as_slice().first(), steps.as_slice().get(1))
    else {
        return false;
    };
    let (ObjectIndexSelector::Member(_), ObjectIndexSelector::IndexValues { .. }) =
        (member_step.selector(), paren_step.selector())
    else {
        return false;
    };
    member_step.kind() == ObjectIndexKind::Member
        && paren_step.kind() == ObjectIndexKind::Paren
        && member_step.origin() == ObjectSubscriptOrigin::DottedInvokeMember
        && paren_step.origin() == ObjectSubscriptOrigin::DottedInvokeArguments
}

// The syntax is ambiguous until member resolution: `obj.data(:)` indexes a
// stored array, whereas a resolved method/function requires value arguments.
fn call_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    components: ExactArrayStorage<ObjectIndexComponent>,
) -> Result<crate::execution::RootedValues, RuntimeError> {
    if components
        .iter()
        .any(|component| matches!(component, ObjectIndexComponent::Colon))
    {
        return Err(semantic_error(
            "InvalidDottedInvokeArgument",
            "colon is not a standalone dotted-invocation argument",
        ));
    }
    super::callable::component_arguments(runtime, components)
}
