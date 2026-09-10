use std::collections::VecDeque;

use runmat_value::Value;

use super::{resolve_single, SubscriptReadRequest};
use crate::object::indexing::{
    ObjectIndexKind, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptOrigin,
};
use crate::runtime_error::semantic_error;
use crate::sequence::ValueSequence;
use crate::RuntimeError;

pub(super) async fn try_execute(
    base: Value,
    steps: &mut VecDeque<ObjectSubscript>,
    request: &SubscriptReadRequest,
    caller_function_name: Option<&str>,
) -> Result<Option<ValueSequence>, RuntimeError> {
    let Some((member, arguments)) = prefix(steps)? else {
        return Ok(None);
    };
    steps.pop_front();
    steps.pop_front();
    let requested_outputs = if steps.is_empty() {
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
        &runmat_types::MethodName::from(member.0.as_str()),
        &request.access,
    )? {
        let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
        call_arguments.push(base);
        call_arguments.extend(arguments);
        crate::object::protocol::invoke_resolved_object_method(
            &method,
            call_arguments,
            requested_outputs,
        )
        .await?
    } else {
        let member_sequence = crate::object::resolve::read_member_sequence_with_context(
            None,
            base,
            member.0,
            false,
            caller_function_name,
        )
        .await?;
        let member_value = resolve_single(member_sequence)?;
        if is_callable_value(&member_value) {
            let resolver = RuntimeFunctionResolver;
            crate::call::descriptor::execute_callable_descriptor(
                crate::call::descriptor::CallableDescriptor::from_feval_value(
                    member_value,
                    arguments,
                    requested_outputs,
                    &resolver,
                ),
            )
            .await?
        } else {
            let step = ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: arguments.into_iter().map(Into::into).collect(),
            });
            super::read_default_step(member_value, &step, caller_function_name).await?
        }
    };
    Ok(Some(ValueSequence::from_callable_result(
        result,
        requested_outputs,
    )))
}

fn prefix(
    steps: &VecDeque<ObjectSubscript>,
) -> Result<Option<(runmat_types::MemberName, Vec<Value>)>, RuntimeError> {
    let (Some(member_step), Some(paren_step)) = (steps.front(), steps.get(1)) else {
        return Ok(None);
    };
    let (ObjectIndexSelector::Member(member), ObjectIndexSelector::IndexValues { components }) =
        (member_step.selector(), paren_step.selector())
    else {
        return Ok(None);
    };
    if member_step.kind() != ObjectIndexKind::Member
        || paren_step.kind() != ObjectIndexKind::Paren
        || member_step.origin() != ObjectSubscriptOrigin::DottedInvokeMember
        || paren_step.origin() != ObjectSubscriptOrigin::DottedInvokeArguments
    {
        return Ok(None);
    }
    let values = components
        .iter()
        .map(|component| match component {
            crate::object::indexing::ObjectIndexComponent::Value(value) => Ok(value.clone()),
            crate::object::indexing::ObjectIndexComponent::Colon => Err(semantic_error(
                "InvalidDottedInvokeArgument",
                "colon is not a standalone dotted-invocation argument",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some((member.clone(), values)))
}

struct RuntimeFunctionResolver;

impl crate::call::descriptor::FunctionNameResolver for RuntimeFunctionResolver {
    fn resolve_function(&self, name: &str) -> Option<runmat_types::FunctionId> {
        crate::user_functions::resolve_semantic_function_by_name(name).map(runmat_types::FunctionId)
    }
}

fn is_callable_value(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}
