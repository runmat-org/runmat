use std::collections::VecDeque;

use runmat_value::Value;

use super::{resolve_single, SubscriptReadRequest};
use crate::execution::RootedValueSequence;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscript,
    ObjectSubscriptOrigin,
};
use crate::runtime_error::semantic_error;
use crate::sequence::ValueSequence;
use crate::RuntimeError;

pub(super) async fn try_execute(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    steps: &mut VecDeque<ObjectSubscript>,
    request: &SubscriptReadRequest,
    caller_function_name: Option<&str>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let Some((member, components, semantic_fallback)) = prefix(steps) else {
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
        let arguments = call_arguments(components)?;
        let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
        call_arguments.push(base);
        call_arguments.extend(arguments);
        match runtime {
            Some(runtime) => {
                crate::object::protocol::invoke_resolved_object_method_in_context(
                    runtime,
                    &method,
                    call_arguments,
                    requested_outputs,
                )
                .await?
            }
            None => {
                crate::object::protocol::invoke_resolved_object_method(
                    &method,
                    call_arguments,
                    requested_outputs,
                )
                .await?
            }
        }
    } else if crate::object::resolve::has_stored_member(&base, &member)?
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
        let arguments = call_arguments(components)?;
        match runtime {
            Some(runtime) => {
                crate::object::dispatch::call_method_or_member_index_with_fallback_in_context(
                    runtime,
                    base,
                    member.0,
                    arguments,
                    requested_outputs,
                    caller_function_name,
                    semantic_fallback,
                )
                .await?
            }
            None => {
                crate::object::dispatch::call_method_or_member_index_with_fallback(
                    base,
                    member.0,
                    arguments,
                    requested_outputs,
                    caller_function_name,
                    semantic_fallback,
                )
                .await?
            }
        }
    } else {
        let member_sequence = crate::object::resolve::read_member_sequence_with_context(
            runtime,
            base,
            member.0,
            false,
            caller_function_name,
        )
        .await?;
        let member_value = resolve_single(member_sequence)?;
        if is_callable_value(&member_value) {
            let arguments = call_arguments(components)?;
            let resolver = RuntimeFunctionResolver;
            let descriptor = crate::call::descriptor::CallableDescriptor::from_feval_value(
                member_value.value().clone(),
                arguments,
                requested_outputs,
                &resolver,
            );
            match runtime {
                Some(runtime) => {
                    crate::call::descriptor::execute_callable_descriptor_in_context(
                        runtime, descriptor,
                    )
                    .await?
                }
                None => crate::call::descriptor::execute_callable_descriptor(descriptor).await?,
            }
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
            .try_map_owned(
                |value| {
                    ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            )?
        }
    };
    Ok(Some(result))
}

fn prefix(
    steps: &VecDeque<ObjectSubscript>,
) -> Option<(
    runmat_types::MemberName,
    Vec<ObjectIndexComponent>,
    Option<runmat_types::CallableIdentity>,
)> {
    let (Some(member_step), Some(paren_step)) = (steps.front(), steps.get(1)) else {
        return None;
    };
    let (ObjectIndexSelector::Member(member), ObjectIndexSelector::IndexValues { components }) =
        (member_step.selector(), paren_step.selector())
    else {
        return None;
    };
    if member_step.kind() != ObjectIndexKind::Member
        || paren_step.kind() != ObjectIndexKind::Paren
        || member_step.origin() != ObjectSubscriptOrigin::DottedInvokeMember
        || paren_step.origin() != ObjectSubscriptOrigin::DottedInvokeArguments
    {
        return None;
    }
    Some((
        member.clone(),
        components.clone(),
        member_step.semantic_fallback().cloned(),
    ))
}

// The syntax is ambiguous until member resolution: `obj.data(:)` indexes a
// stored array, whereas a resolved method/function requires value arguments.
fn call_arguments(components: Vec<ObjectIndexComponent>) -> Result<Vec<Value>, RuntimeError> {
    components
        .into_iter()
        .map(|component| match component {
            ObjectIndexComponent::Value(value) => Ok(value),
            ObjectIndexComponent::Colon => Err(semantic_error(
                "InvalidDottedInvokeArgument",
                "colon is not a standalone dotted-invocation argument",
            )),
        })
        .collect()
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
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}
