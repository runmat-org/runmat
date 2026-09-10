use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

fn write_sequence_register(
    stack: &mut Vec<Value>,
    register: &mut super::SequenceState,
    values: Vec<Value>,
) -> Result<(), RuntimeError> {
    if register.assignment.is_some() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence register was overwritten before consumption",
        ));
    }
    let start = stack.len();
    let len = values.len();
    stack.extend(values);
    register.assignment = Some(super::SequenceRegister { start, len });
    Ok(())
}

pub(super) fn take_sequence_register(
    stack: &mut Vec<Value>,
    register: &mut super::SequenceState,
) -> Result<Vec<Value>, RuntimeError> {
    let window = register
        .assignment
        .take()
        .ok_or(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence assignment has no pending value sequence",
        ))?;
    let end = window
        .start
        .checked_add(window.len)
        .ok_or(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence register bounds overflowed",
        ))?;
    if end != stack.len() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence register did not remain the top operand-stack window",
        ));
    }
    Ok(stack.split_off(window.start))
}

pub struct ObjectDispatchContext<'a> {
    pub vars: &'a [Value],
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub current_function_name: &'a str,
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub call_counts: &'a [(usize, usize)],
}

pub async fn dispatch_object(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    sequence_register: &mut super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    let ObjectDispatchContext {
        vars,
        runtime,
        current_function_name,
        bytecode,
        call_counts,
    } = context;
    let caller_function_name = if current_function_name.is_empty() {
        None
    } else {
        Some(current_function_name)
    };
    match instr {
        crate::bytecode::Instr::BeginSubscriptEndReceiver { prefix } => {
            let operand_count = prefix
                .iter()
                .map(crate::bytecode::BytecodeSubscriptStep::operand_count)
                .sum::<usize>();
            let start = stack.len().checked_sub(operand_count + 1).ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow")
            })?;
            let (root, path) = materialize_subscript_path(stack[start..].to_vec(), prefix)?;
            let prepared = runmat_runtime::object::protocol::prepare_subscript_receiver(
                root,
                path,
                frame_access(bytecode)?,
            )
            .await?;
            let stack_index = stack.len();
            stack.push(prepared.receiver().clone());
            sequence_register.push_subscript_receiver(prepared, start, stack_index);
            Ok(true)
        }
        crate::bytecode::Instr::LoadSubscriptEnd {
            component,
            component_count,
        } => {
            let value = runmat_runtime::object::protocol::resolve_subscript_end(
                sequence_register.subscript_receiver()?,
                *component,
                *component_count,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::FinishSubscriptEndReceiver {
            selector_count,
            prefix_operand_count,
        } => {
            let (prepared, prefix_start, expected_receiver) =
                sequence_register.pop_subscript_receiver()?;
            let receiver = stack.len().checked_sub(selector_count + 1).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "subscript receiver stack underflow",
                )
            })?;
            if receiver != expected_receiver {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "subscript receiver root moved before completion",
                ));
            }
            if receiver.checked_sub(prefix_start) != Some(*prefix_operand_count + 1) {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "prepared subscript prefix operand count is inconsistent",
                ));
            }
            stack.remove(receiver);
            if prefix_start > receiver {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "subscript prefix root follows its prepared receiver",
                ));
            }
            stack.drain(prefix_start..receiver);
            stack.insert(prefix_start, prepared.receiver().clone());
            Ok(true)
        }
        crate::bytecode::Instr::ReadSubscriptPath {
            steps,
            selection,
            context,
            to_sequence_register,
        } => {
            let values = execute_subscript_path(
                stack,
                steps,
                *selection,
                *context,
                frame_access(bytecode)?,
                call_counts,
                sequence_register,
            )
            .await?;
            if *to_sequence_register {
                write_sequence_register(stack, sequence_register, values)?;
            } else {
                stack.extend(values);
            }
            Ok(true)
        }
        crate::bytecode::Instr::CaptureSubscriptPath {
            steps,
            sequence_slot,
            context,
        } => {
            let values = execute_subscript_path(
                stack,
                steps,
                runmat_types::SequenceUse::ExpandAll,
                *context,
                frame_access(bytecode)?,
                call_counts,
                sequence_register,
            )
            .await?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::LoadMember(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberOrInit(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamic => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberSequence { member, selection } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
            )
            .await?
            .resolve(*selection, Default::default())?;
            stack.extend(values);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicSequence { selection } => {
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?
            .resolve(*selection, Default::default())?;
            stack.extend(values);
            Ok(true)
        }
        crate::bytecode::Instr::MemberSequenceCardinality => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let count = obj_resolve::member_sequence_cardinality(&base)?;
            stack.push(Value::Num(count as f64));
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot {
            member,
            output_count_slot,
        } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let count = super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let sequence = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            let values = sequence.resolve(
                runmat_types::SequenceUse::SelectPrefix { count },
                Default::default(),
            )?;
            write_sequence_register(stack, sequence_register, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { output_count_slot } => {
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let count = super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let sequence = obj_resolve::read_member_sequence_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            let values = sequence.resolve(
                runmat_types::SequenceUse::SelectPrefix { count },
                Default::default(),
            )?;
            write_sequence_register(stack, sequence_register, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureCallOutputSequence => {
            let legacy = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = match legacy {
                Value::OutputList(values) => values,
                value => vec![value],
            };
            write_sequence_register(stack, sequence_register, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureScalarSequence => {
            let value = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            write_sequence_register(stack, sequence_register, vec![value])?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureMemberSequence {
            member,
            sequence_slot,
        } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::Member {
                    base,
                    member: member.clone(),
                },
            )
            .await?
            .resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureMemberDynamicSequence { sequence_slot } => {
            let member = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::DynamicMember {
                    base,
                    member,
                },
            )
            .await?
            .resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureCellContentsSequence {
            sequence_slot,
            num_indices,
            expand_all,
        } => {
            let mut indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::CellContents {
                    base,
                    indices,
                    expand_all: *expand_all,
                },
            )
            .await?
            .resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureReturnedOutputsSequence { sequence_slot } => {
            let value = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::ReturnedOutputs(
                    value,
                ),
            )
            .await?
            .resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMember(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::store_member_traced(
                base,
                field.0.clone(),
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberOrInit(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::store_member_traced(
                base,
                field.0.clone(),
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamic => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced(
                base,
                name,
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicOrInit => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced(
                base,
                name,
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberSequence(field) => {
            let values = take_sequence_register(stack, sequence_register)?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let updated = obj_resolve::store_member_sequence_traced(
                base,
                field.0.clone(),
                values,
                caller_function_name,
            )
            .await?;
            stack.push(updated);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicSequence => {
            let values = take_sequence_register(stack, sequence_register)?;
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let updated =
                obj_resolve::store_member_sequence_traced(base, name, values, caller_function_name)
                    .await?;
            stack.push(updated);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn frame_access(
    bytecode: &crate::bytecode::Bytecode,
) -> Result<runmat_runtime::object::protocol::ObjectAccessContext, RuntimeError> {
    match bytecode.active_function {
        Some(function) => runmat_runtime::object::protocol::ObjectAccessContext::from_method_frame(
            bytecode.active_class_method_owner.as_ref(),
            &runmat_types::CallableIdentity::BoundFunction(function),
        ),
        None => Ok(Default::default()),
    }
}

async fn execute_subscript_path(
    stack: &mut Vec<Value>,
    steps: &[crate::bytecode::BytecodeSubscriptStep],
    selection: runmat_types::SequenceUse,
    indexing_context: runmat_types::ObjectIndexingContext,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
    call_counts: &[(usize, usize)],
    state: &super::SequenceState,
) -> Result<Vec<Value>, RuntimeError> {
    let operand_count = steps
        .iter()
        .map(crate::bytecode::BytecodeSubscriptStep::operand_count)
        .sum::<usize>();
    let start = stack.len().checked_sub(operand_count + 1).ok_or_else(|| {
        crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow")
    })?;
    let (root, path) = materialize_subscript_path(stack.split_off(start), steps)?;
    let path = path.ok_or_else(|| {
        crate::interpreter::errors::mex(
            "RunMat:InvalidObjectSubscriptPath",
            "subscript path is empty",
        )
    })?;
    let sequence_context = match selection {
        runmat_types::SequenceUse::SelectCurrentFunctionOutputs => {
            runmat_runtime::sequence::SequenceResolutionContext::current_function_outputs(
                call_counts.last().map_or(1, |entry| entry.1),
            )
        }
        runmat_types::SequenceUse::SelectDestinationCardinality => {
            runmat_runtime::sequence::SequenceResolutionContext::destination_cardinality(
                state.output_layout()?.total(),
            )
        }
        _ => Default::default(),
    };
    let request = runmat_runtime::object::protocol::SubscriptReadRequest {
        sequence_use: selection,
        sequence_context,
        indexing_context,
        access,
    };
    let sequence = runmat_runtime::object::protocol::read_subscript_path_sequence_with_access(
        root, path, request,
    )
    .await?;
    sequence.resolve(selection, sequence_context)
}

fn materialize_subscript_path(
    values: Vec<Value>,
    steps: &[crate::bytecode::BytecodeSubscriptStep],
) -> Result<
    (
        Value,
        Option<runmat_runtime::object::indexing::ObjectSubscriptPath>,
    ),
    RuntimeError,
> {
    use crate::bytecode::BytecodeSubscriptStep as Step;
    use runmat_runtime::object::indexing::{
        ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
    };
    let mut values = values.into_iter();
    let root = values.next().ok_or_else(|| {
        crate::interpreter::errors::mex("StackUnderflow", "subscript path has no root")
    })?;
    let mut runtime_steps = Vec::new();
    for step in steps {
        match step {
            Step::Member(member) => runtime_steps.push(ObjectSubscript::member(member.clone())),
            Step::DynamicMember => {
                let value = values.next().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "dynamic member is missing")
                })?;
                runtime_steps.push(ObjectSubscript::member(String::try_from(&value)?));
            }
            Step::Parentheses { selectors } | Step::Braces { selectors } => {
                let components = materialize_selector_components(&mut values, selectors)?;
                let selector = ObjectIndexSelector::IndexValues { components };
                runtime_steps.push(if matches!(step, Step::Parentheses { .. }) {
                    ObjectSubscript::parentheses(selector)
                } else {
                    ObjectSubscript::braces(selector)
                });
            }
            Step::DottedInvoke { member, arguments } => {
                let components = materialize_selector_components(&mut values, arguments)?;
                runtime_steps.extend(ObjectSubscript::dotted_invoke_components(
                    member.clone(),
                    components,
                ));
            }
        }
    }
    if values.next().is_some() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:SubscriptPathState",
            "subscript path has excess operands",
        ));
    }
    let path = if runtime_steps.is_empty() {
        None
    } else {
        Some(ObjectSubscriptPath::new(runtime_steps)?)
    };
    Ok((root, path))
}

fn materialize_selector_components(
    values: &mut impl Iterator<Item = Value>,
    selectors: &[crate::bytecode::BytecodeSubscriptSelector],
) -> Result<Vec<runmat_runtime::object::indexing::ObjectIndexComponent>, RuntimeError> {
    use crate::bytecode::BytecodeSubscriptSelector as Selector;
    use runmat_runtime::object::indexing::ObjectIndexComponent;
    selectors
        .iter()
        .map(|selector| match selector {
            Selector::Colon => Ok(ObjectIndexComponent::Colon),
            Selector::Value => values
                .next()
                .map(ObjectIndexComponent::Value)
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "StackUnderflow",
                        "subscript selector is missing",
                    )
                }),
        })
        .collect()
}
