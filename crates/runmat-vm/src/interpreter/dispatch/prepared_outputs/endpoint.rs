use super::support::{finish_destination, pop_selector_components, stack_underflow};
use super::{admit, PreparedDispatchFuture, PreparedResult, Request};
use crate::bytecode::Instr;
use futures::FutureExt;
use runmat_runtime::sequence::AssignmentStepSpec;
use runmat_runtime::sequence::SequenceEndpointSpec;
use runmat_runtime::RuntimeError;

#[inline(never)]
pub(super) fn finish_member_sequence_output_target<'a>(
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
                Instr::FinishMemberSequenceOutputTarget(member) => {
                    let member = runmat_runtime::call::argument_text::copy_str(
                        &member.0,
                        Some(&context.authority.context.runtime),
                    )?;
                    finish_destination(
                        state,
                        &context.authority.context.runtime,
                        SequenceEndpointSpec::Member(member),
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
pub(super) fn finish_fixed_assignment_target<'a>(
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
                Instr::FinishFixedAssignmentTarget {
                    endpoint,
                    selector_context,
                    mutation,
                } => {
                    if state.fixed_assignment_owner.is_none() {
                        return Err(crate::interpreter::errors::mex(
                            "FixedAssignmentOwner",
                            "fixed destination finishes without its preparation",
                        ));
                    }
                    use crate::bytecode::BytecodeSubscriptStep as Step;
                    let mut retained = None;
                    let step = match endpoint {
                        Step::Member(member) => AssignmentStepSpec::Member(
                            runmat_runtime::call::argument_text::copy_str(
                                &member.0,
                                Some(&context.authority.context.runtime),
                            )?,
                        ),
                        Step::DynamicMember => AssignmentStepSpec::Member(
                            runmat_runtime::call::argument_text::convert(
                                &stack.pop().ok_or_else(stack_underflow)?,
                                Some(&context.authority.context.runtime),
                            )?,
                        ),
                        Step::Parentheses { selectors } | Step::Braces { selectors } => {
                            let (selectors, roots) = pop_selector_components(
                                stack,
                                selectors,
                                false,
                                &context.authority.context.runtime,
                                state,
                            )
                            .await?;
                            retained = Some(roots);
                            if matches!(endpoint, Step::Parentheses { .. }) {
                                AssignmentStepSpec::Parentheses { selectors }
                            } else {
                                AssignmentStepSpec::Braces(selectors)
                            }
                        }
                        Step::DottedInvoke { .. } => {
                            return Err(crate::interpreter::errors::mex(
                                "InvalidAssignmentTarget",
                                "invocation cannot be an assignment endpoint",
                            ))
                        }
                    };
                    if let Some(count) = selector_context {
                        state.finish_index_selectors(*count)?;
                    }
                    finish_destination(
                        state,
                        &context.authority.context.runtime,
                        SequenceEndpointSpec::Fixed {
                            step,
                            mutation: *mutation,
                        },
                    )
                    .await?;
                    drop(retained);
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn finish_dynamic_member_sequence_output_target<'a>(
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
                Instr::FinishDynamicMemberSequenceOutputTarget => {
                    let member = runmat_runtime::call::argument_text::convert(
                        &stack.pop().ok_or_else(stack_underflow)?,
                        Some(&context.authority.context.runtime),
                    )?;
                    finish_destination(
                        state,
                        &context.authority.context.runtime,
                        SequenceEndpointSpec::Member(member),
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
pub(super) fn finish_cell_contents_sequence_output_target<'a>(
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
                Instr::FinishCellContentsSequenceOutputTarget {
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
                    finish_destination(
                        state,
                        &context.authority.context.runtime,
                        SequenceEndpointSpec::CellContents(indices),
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
