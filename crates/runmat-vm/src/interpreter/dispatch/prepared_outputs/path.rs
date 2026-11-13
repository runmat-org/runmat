use super::support::{pop_selector_components, stack_underflow};
use super::{admit, PreparedDispatchFuture, PreparedResult, Request};
use crate::bytecode::Instr;
use futures::FutureExt;
use runmat_runtime::sequence::AssignmentStepSpec;
use runmat_runtime::RuntimeError;

#[inline(never)]
pub(super) fn prepare_member_path_step<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::PrepareMemberPathStep {
                    member,
                    child_creation,
                } => {
                    state
                        .destination_builder_mut()?
                        .push_assignment_step_in_context(
                            &context.authority.context.runtime,
                            AssignmentStepSpec::Member(
                                runmat_runtime::call::argument_text::copy_str(
                                    &member.0,
                                    Some(&context.authority.context.runtime),
                                )?,
                            ),
                            *child_creation,
                            Some(context.authority.current_function_name),
                        )
                        .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn prepare_dynamic_member_path_step<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::PrepareDynamicMemberPathStep { child_creation } => {
                    let member = runmat_runtime::call::argument_text::convert(
                        &stack.pop().ok_or_else(stack_underflow)?,
                        Some(&context.authority.context.runtime),
                    )?;
                    state
                        .destination_builder_mut()?
                        .push_assignment_step_in_context(
                            &context.authority.context.runtime,
                            AssignmentStepSpec::Member(member),
                            *child_creation,
                            Some(context.authority.current_function_name),
                        )
                        .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn prepare_parentheses_path_step<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::PrepareParenthesesPathStep {
                    child_creation,
                    component_count,
                    selectors,
                } => {
                    let (selectors, _selector_roots) = pop_selector_components(
                        stack,
                        selectors,
                        false,
                        &context.authority.context.runtime,
                        state,
                    )
                    .await?;
                    state.finish_index_selectors(*component_count)?;
                    state
                        .destination_builder_mut()?
                        .push_assignment_step_in_context(
                            &context.authority.context.runtime,
                            AssignmentStepSpec::Parentheses { selectors },
                            *child_creation,
                            Some(context.authority.current_function_name),
                        )
                        .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn prepare_braces_path_step<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::PrepareBracesPathStep {
                    child_creation,
                    component_count,
                    selectors,
                    expand_all,
                } => {
                    let (indices, _selector_roots) = pop_selector_components(
                        stack,
                        selectors,
                        *expand_all,
                        &context.authority.context.runtime,
                        state,
                    )
                    .await?;
                    state.finish_index_selectors(*component_count)?;
                    state
                        .destination_builder_mut()?
                        .push_assignment_step_in_context(
                            &context.authority.context.runtime,
                            AssignmentStepSpec::Braces(indices),
                            *child_creation,
                            Some(context.authority.current_function_name),
                        )
                        .await?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}
