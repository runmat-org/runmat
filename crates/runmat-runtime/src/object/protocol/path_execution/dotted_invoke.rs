use runmat_value::{ExactArrayIntoIter, ExactArrayStorage, Value};

use super::{resolve_single, SubscriptReadRequest};
use crate::execution::RootedValueSequence;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexKind, ObjectIndexSelector, ObjectSubscript,
    ObjectSubscriptOrigin,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use futures::{future::Either, FutureExt};

enum SelectedCall {
    Declared {
        method: super::super::ResolvedObjectMethod,
        arguments: crate::call::BuiltinArguments,
    },
    Fallback {
        base: Value,
        member: runmat_value::Utf8Text,
        arguments: crate::call::BuiltinArguments,
        semantic: Option<std::sync::Arc<runmat_types::CallableIdentity>>,
    },
    Stored(StoredMemberSelection),
}

struct StoredMemberSelection {
    base: Value,
    member: runmat_value::Utf8Text,
    components: ExactArrayStorage<ObjectIndexComponent>,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
}

pub(super) fn try_execute<'a, A: std::borrow::Borrow<super::super::ObjectAccessContext> + 'a>(
    runtime: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    steps: &'a mut ExactArrayIntoIter<ObjectSubscript>,
    request: &'a SubscriptReadRequest<A>,
    caller_function_name: Option<&'a str>,
) -> impl std::future::Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + 'a {
    // A miss only observes the borrowed selectors: do not allocate a dotted
    // invocation owner for ordinary indexing. Selection retains the same
    // mutable borrow, so no caller can change this prefix before polling.
    if !is_prefix(steps) {
        return futures::future::Either::Left(std::future::ready(Ok(None)));
    }
    // Own the selected dotted operation with canonical path admission.
    // Its factory is quoted before construction, keeping
    // this large child out of the enclosing default-path poll frame. Step
    // consumption, resolution, arguments, and effects remain poll-deferred.
    futures::future::Either::Right(super::super::context::admit_future_with(
        runtime,
        move || {
            crate::execution::after_result(
                async move {
                    let member_step = steps.next().ok_or_else(|| {
                        semantic_error(
                            "InvalidObjectSubscriptPath",
                            "dotted invocation has no member",
                        )
                    })?;
                    let (ObjectIndexSelector::Member(member), semantic_fallback) =
                        member_step.into_parts()
                    else {
                        unreachable!("validated dotted invocation has a member selector")
                    };
                    let paren_step = steps.next().ok_or_else(|| {
                        semantic_error(
                            "InvalidObjectSubscriptPath",
                            "dotted invocation has no arguments",
                        )
                    })?;
                    let input_callsite = paren_step.input_callsite().cloned();
                    let (ObjectIndexSelector::IndexValues { components }, _) =
                        paren_step.into_parts()
                    else {
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
                    let selected = if let Some(method) =
                        crate::object::protocol::resolve_declared_object_method(
                            &base,
                            &runmat_types::MethodName::from(member.as_str()),
                            request.access(),
                        )? {
                        let arguments: crate::call::BuiltinArguments =
                            call_arguments(runtime, components)?.into();
                        let arguments = arguments.try_insert(
                            0,
                            crate::execution::RootedValue::in_optional_context(base, runtime)?,
                            runtime,
                        )?;
                        SelectedCall::Declared { method, arguments }
                    } else if matches!(&base, Value::Foreign(reference)
                        if crate::foreign::dotted_member_role(reference, member.as_str())
                            == crate::foreign::ForeignDottedMemberRole::ReservedMethod)
                    {
                        // Only an exact adapter-owned reserved method skips
                        // property lookup. Unknown foreign members and stored
                        // fields retain their existing read/index semantics.
                        let arguments = call_arguments(runtime, components)?;
                        SelectedCall::Fallback {
                            base,
                            member,
                            arguments: arguments.into(),
                            semantic: None,
                        }
                    } else if crate::object::resolve::has_stored_member(
                        &base,
                        &member.as_str().into(),
                    )? == crate::object::resolve::MemberPresence::Absent
                        && (matches!(
                            &base,
                            Value::Object(_)
                                | Value::ObjectArray(_)
                                | Value::HandleObject(_)
                                | Value::ClassRef(_)
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
                        SelectedCall::Fallback {
                            base,
                            member,
                            arguments: arguments.into(),
                            semantic: semantic_fallback,
                        }
                    } else {
                        SelectedCall::Stored(StoredMemberSelection {
                            base,
                            member,
                            components,
                            input_callsite,
                        })
                    };
                    Ok((selected, requested_outputs))
                },
                move |(selected, requested_outputs)| match selected {
                    SelectedCall::Declared { method, arguments } => Either::Left(async move {
                        crate::object::protocol::invoke_resolved_object_method_retained(
                            runtime,
                            &method,
                            arguments,
                            requested_outputs,
                        )
                        .await
                        .map(Some)
                    }),
                    SelectedCall::Fallback {
                        base,
                        member,
                        arguments,
                        semantic,
                    } => Either::Right(Either::Left(
                        crate::object::dispatch::call_method_or_member_index_retained_with_context(
                            runtime,
                            base,
                            member,
                            arguments,
                            requested_outputs,
                            caller_function_name,
                            semantic,
                        )
                        .map(|result| result.map(Some)),
                    )),
                    SelectedCall::Stored(selected) => Either::Right(Either::Right(stored(
                        runtime,
                        selected,
                        request,
                        requested_outputs,
                        caller_function_name,
                    ))),
                },
            )
        },
    ))
}

async fn stored<A: std::borrow::Borrow<super::super::ObjectAccessContext>>(
    runtime: Option<&crate::context::RuntimeContext>,
    selected: StoredMemberSelection,
    request: &SubscriptReadRequest<A>,
    requested_outputs: usize,
    caller_function_name: Option<&str>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let StoredMemberSelection {
        base,
        member,
        components,
        input_callsite,
    } = selected;
    let member_sequence =
        crate::object::resolve::read_default_member_sequence_text_with_optional_access(
            runtime,
            base,
            member,
            false,
            caller_function_name,
            Some(request.access()),
        )
        .await?;
    let member_value = resolve_single(member_sequence)?;
    let result = if super::callable::is_callable_value(&member_value) {
        let arguments = call_arguments(runtime, components)?;
        super::callable::invoke(
            runtime,
            member_value.value().clone(),
            arguments.into(),
            requested_outputs,
            input_callsite,
        )
        .await?
    } else {
        let step = ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues { components })
            .with_input_callsite(input_callsite);
        let path = crate::object::indexing::ObjectSubscriptPath::single_in_context(runtime, step)?;
        let child_request = SubscriptReadRequest {
            sequence_use: if requested_outputs == 0 {
                runmat_types::SequenceUse::Discard
            } else {
                runmat_types::SequenceUse::SelectPrefix {
                    count: requested_outputs,
                }
            },
            sequence_context: crate::sequence::SequenceResolutionContext::default(),
            indexing_context: request.indexing_context,
            access: request.access(),
        };
        super::read_subscript_path_sequence(
            runtime.cloned(),
            member_value.value().clone(),
            path,
            child_request,
            caller_function_name,
        )
        .await?
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
        && matches!(
            (member_step.origin(), paren_step.origin()),
            (
                ObjectSubscriptOrigin::DottedInvokeMember,
                ObjectSubscriptOrigin::DottedInvokeArguments
            ) | (
                ObjectSubscriptOrigin::Ordinary,
                ObjectSubscriptOrigin::Ordinary
            )
        )
}

// The syntax is ambiguous until member resolution: `obj.data(:)` indexes a
// stored array, whereas a resolved method/function requires value arguments.
fn call_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    components: ExactArrayStorage<ObjectIndexComponent>,
) -> Result<crate::execution::RootedValues, RuntimeError> {
    if components.iter().any(|component| {
        matches!(
            component,
            ObjectIndexComponent::Colon | ObjectIndexComponent::End
        )
    }) {
        return Err(semantic_error(
            "InvalidDottedInvokeArgument",
            "colon is not a standalone dotted-invocation argument",
        ));
    }
    super::callable::component_arguments(runtime, components)
}
