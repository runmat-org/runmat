use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::sequence::{AssignmentStepSpec, SequenceEndpointSpec};
use runmat_runtime::RuntimeError;
use runmat_value::{IntValue, Value};

use super::{PreparedOutputTarget, SequenceState};

mod commit;
mod legacy;
mod support;
use support::{
    empty_cell_root, finish_destination, pop_selector_components, root, stack_underflow,
};

pub(super) struct PreparedOutputContext<'a> {
    pub authority: super::slot_authority::AssignmentSlotAuthority<'a>,
}

pub(super) async fn dispatch(
    instruction: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &mut InterpreterVars,
    state: &mut SequenceState,
    context: PreparedOutputContext<'_>,
    before_overwrite: &mut dyn FnMut(usize),
    after_store: &mut dyn FnMut(usize),
) -> Result<bool, RuntimeError> {
    if legacy::dispatch(instruction, stack, vars, state).await? {
        return Ok(true);
    }
    use crate::bytecode::Instr;
    match instruction {
        Instr::BeginOutputAssignment { target_count } => {
            state.begin_output_assignment(*target_count)?
        }
        Instr::BeginFixedAssignment {
            source_statement,
            root_slot,
        } => {
            state.begin_output_assignment(1)?;
            state.fixed_assignment_owner = Some((*source_statement, *root_slot));
        }
        Instr::PrepareFixedOutputTarget => state.push_output_target(PreparedOutputTarget::Fixed)?,
        Instr::PrepareDiscardOutputTarget => {
            state.push_output_target(PreparedOutputTarget::Discard)?
        }
        Instr::BeginSequenceOutputTarget {
            root_slot,
            root_creation,
        } => {
            match state.fixed_assignment_owner {
                Some((_, expected))
                    if expected == *root_slot && root_creation.retains_original_receiver() => {}
                None if !root_creation.retains_original_receiver() => {}
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "FixedAssignmentOwner",
                        "destination root disagrees with its fixed preparation",
                    ))
                }
            }
            let root_presence = context.authority.refresh(vars, *root_slot)?;
            let absent_root = if root_presence.is_absent() {
                use runmat_mir::SequenceRootCreation as Creation;
                use runmat_types::AssignmentCreationPolicy as Policy;
                match root_creation {
                    Creation::EmptyCell | Creation::OrdinaryCell => Some(empty_cell_root()),
                    Creation::Ordinary(Policy::CreateArrayByIndex) => Some(Value::Tensor(
                        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(|error| {
                            crate::interpreter::errors::mex("IndexAssignmentBaseInit", &error)
                        })?,
                    )),
                    Creation::Ordinary(Policy::CreateStructFieldPath) => {
                        Some(Value::Struct(runmat_value::StructValue::new()))
                    }
                    Creation::Ordinary(Policy::ExistingOnly) => {
                        return Err(crate::interpreter::errors::mex(
                            "UndefinedVariable",
                            "deletion requires an existing assignment root",
                        ))
                    }
                    Creation::Ordinary(_) => {
                        return Err(crate::interpreter::errors::mex(
                            "InvalidAssignmentCreationPolicy",
                            "unsupported ordinary assignment root creation policy",
                        ))
                    }
                    Creation::Existing => None,
                }
            } else {
                None
            };
            let original = root(vars, *root_slot)?;
            match absent_root {
                Some(seed) => state.begin_destination(
                    &context.authority.context.runtime,
                    *root_slot,
                    &seed,
                    Some(seed.clone()),
                    *root_creation,
                )?,
                None => state.begin_destination(
                    &context.authority.context.runtime,
                    *root_slot,
                    &original,
                    None,
                    *root_creation,
                )?,
            }
        }
        Instr::PrepareMemberPathStep(member) => {
            state
                .destination_builder_mut()?
                .push_step_in_context(
                    &context.authority.context.runtime,
                    AssignmentStepSpec::Member(runmat_runtime::call::argument_text::copy_str(
                        &member.0,
                        Some(&context.authority.context.runtime),
                    )?),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::PrepareDynamicMemberPathStep => {
            let member = runmat_runtime::call::argument_text::convert(
                &stack.pop().ok_or_else(stack_underflow)?,
                Some(&context.authority.context.runtime),
            )?;
            state
                .destination_builder_mut()?
                .push_step_in_context(
                    &context.authority.context.runtime,
                    AssignmentStepSpec::Member(member),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::BeginPreparedIndexSelectors { component_count } => {
            state.begin_index_selectors(*component_count)?;
        }
        Instr::LoadPreparedIndexEnd { component } => {
            let component_count = state.index_selector_count()?;
            if state.prepared_index_receiver.is_none() {
                let prepared = state
                    .destination_builder_mut()?
                    .prepare_current_receiver_for_end_in_context(
                        &context.authority.context.runtime,
                        Some(context.authority.current_function_name),
                    )
                    .await?;
                state.prepared_index_receiver = Some(prepared);
            }
            let receiver = state
                .prepared_index_receiver
                .as_ref()
                .ok_or_else(stack_underflow)?;
            let value = runmat_runtime::object::protocol::resolve_subscript_end_in_context(
                &context.authority.context.runtime,
                receiver,
                *component,
                component_count,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
        }
        Instr::BeginContextualIndexSelectors { component_count } => {
            let base = stack.last().ok_or_else(stack_underflow)?;
            let shape = runmat_runtime::builtins::common::shape::value_dimensions_borrowed(base)?;
            state.begin_contextual_index(*component_count, shape.as_slice())?;
        }
        Instr::LoadContextualIndexEnd { component } => {
            let extent = state.contextual_index_extent(*component)?;
            stack.try_push(Value::Num(extent as f64))?;
        }
        Instr::FinishContextualIndexSelectors { component_count } => {
            state.finish_contextual_index(*component_count)?;
        }
        Instr::PrepareParenthesesPathStep {
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
                .push_step_in_context(
                    &context.authority.context.runtime,
                    AssignmentStepSpec::Parentheses { selectors },
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::PrepareBracesPathStep {
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
                .push_step_in_context(
                    &context.authority.context.runtime,
                    AssignmentStepSpec::Braces(indices),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
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
                Step::Member(member) => {
                    AssignmentStepSpec::Member(runmat_runtime::call::argument_text::copy_str(
                        &member.0,
                        Some(&context.authority.context.runtime),
                    )?)
                }
                Step::DynamicMember => {
                    AssignmentStepSpec::Member(runmat_runtime::call::argument_text::convert(
                        &stack.pop().ok_or_else(stack_underflow)?,
                        Some(&context.authority.context.runtime),
                    )?)
                }
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
        Instr::LoadPreparedOutputCardinality => {
            let total = u64::try_from(state.output_cardinality()?).map_err(|_| {
                crate::interpreter::errors::mex(
                    "DestinationCardinalityOverflow",
                    "output destination cardinality exceeds the supported limit",
                )
            })?;
            stack.try_push(Value::Int(IntValue::U64(total)))?;
        }
        Instr::CommitPreparedOutputTargets { retained_outputs } => {
            if state.fixed_assignment_owner.is_some() {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:PreparedAssignmentOwner",
                    "fixed assignment requires its exact typed consumer",
                ));
            }
            commit::commit(
                stack,
                vars,
                state,
                *retained_outputs,
                &context.authority,
                before_overwrite,
                after_store,
            )
            .await?;
        }
        Instr::CommitFixedAssignment {
            source_statement,
            root_slot,
        } => {
            if state.fixed_assignment_owner != Some((*source_statement, *root_slot)) {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:PreparedAssignmentOwner",
                    "fixed assignment consumer differs from its source occurrence",
                ));
            }
            commit::commit_fixed(
                stack,
                vars,
                state,
                &context.authority,
                before_overwrite,
                after_store,
            )
            .await?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
