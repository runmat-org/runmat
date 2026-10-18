use runmat_runtime::execution::RootedValues;
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

mod capture;
mod read;
mod subscript;
mod write;

pub(super) fn write_sequence_register(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    register: &mut super::SequenceState,
    values: impl Into<runmat_value::OutputListStorage>,
) -> Result<(), RuntimeError> {
    if register.assignment.is_some() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence register was overwritten before consumption",
        ));
    }
    let values = values.into();
    let start = stack.len();
    let len = values.len();
    stack.try_extend_output(values)?;
    register.assignment = Some(super::SequenceRegister { start, len });
    Ok(())
}

pub(super) fn write_rooted_sequence_register(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    register: &mut super::SequenceState,
    values: RootedValues,
) -> Result<(), RuntimeError> {
    values.with_owned_output_storage(|values| {
        if register.assignment.is_some() {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "sequence register was overwritten before consumption",
            ));
        }
        let start = stack.len();
        let len = values.as_slice().len();
        stack.try_extend_output(values)?;
        register.assignment = Some(super::SequenceRegister { start, len });
        Ok(())
    })
}

pub(super) fn take_sequence_register(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    register: &mut super::SequenceState,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
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
    runmat_value::ValueSequence::from_output_storage(stack.try_extract_output(window.start)?)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
}

pub struct ObjectDispatchContext<'a> {
    pub vars: &'a crate::interpreter::vars::InterpreterVars,
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub current_function_name: &'a str,
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
}

pub async fn dispatch_object(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_register: &mut super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::BeginSubscriptEndReceiver { .. }
        | crate::bytecode::Instr::LoadSubscriptEnd { .. }
        | crate::bytecode::Instr::FinishSubscriptEndReceiver { .. }
        | crate::bytecode::Instr::FinishDottedInvokeReceiver { .. }
        | crate::bytecode::Instr::ReadSubscriptPath { .. }
        | crate::bytecode::Instr::CaptureSubscriptPath { .. } => {
            subscript::dispatch(instr, stack, sequence_register, context).await
        }
        crate::bytecode::Instr::LoadMember(_)
        | crate::bytecode::Instr::LoadMemberOrInit(_)
        | crate::bytecode::Instr::LoadMemberDynamic
        | crate::bytecode::Instr::LoadMemberDynamicOrInit
        | crate::bytecode::Instr::LoadMemberSequence { .. }
        | crate::bytecode::Instr::LoadMemberDynamicSequence { .. }
        | crate::bytecode::Instr::MemberSequenceCardinality
        | crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot { .. }
        | crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. } => {
            read::dispatch(instr, stack, sequence_register, context).await
        }
        crate::bytecode::Instr::CaptureCallOutputSequence
        | crate::bytecode::Instr::CaptureScalarSequence
        | crate::bytecode::Instr::CaptureMemberSequence { .. }
        | crate::bytecode::Instr::CaptureMemberDynamicSequence { .. }
        | crate::bytecode::Instr::CaptureCellContentsSequence { .. }
        | crate::bytecode::Instr::CaptureReturnedOutputsSequence { .. } => {
            capture::dispatch(instr, stack, sequence_register, context).await
        }
        crate::bytecode::Instr::StoreMember(_)
        | crate::bytecode::Instr::StoreMemberOrInit(_)
        | crate::bytecode::Instr::StoreMemberDynamic
        | crate::bytecode::Instr::StoreMemberDynamicOrInit
        | crate::bytecode::Instr::StoreMemberSequence(_)
        | crate::bytecode::Instr::StoreMemberDynamicSequence => {
            write::dispatch(instr, stack, sequence_register, context).await
        }
        _ => Ok(false),
    }
}

fn frame_access(
    bytecode: &crate::bytecode::Bytecode,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::object::protocol::ObjectAccessContext, RuntimeError> {
    match bytecode.active_function {
        Some(_)
            if runtime.selected_object_class().is_some()
                && bytecode.active_class_method_owner.is_some() =>
        {
            runmat_runtime::object::protocol::ObjectAccessContext::from_selected_method_frame(
                bytecode
                    .active_class_method_owner
                    .as_ref()
                    .expect("selected method owner checked"),
                runtime
                    .selected_object_class()
                    .expect("selected class facts checked"),
            )
        }
        Some(function) => runmat_runtime::object::protocol::ObjectAccessContext::from_method_frame(
            bytecode.active_class_method_owner.as_ref(),
            &runmat_types::CallableIdentity::BoundFunction(function),
        ),
        None => Ok(Default::default()),
    }
}

struct SubscriptPathReadContext {
    selection: runmat_types::SequenceUse,
    indexing: runmat_types::ObjectIndexingContext,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
    invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
}

async fn execute_subscript_path(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    steps: &[crate::bytecode::BytecodeSubscriptStep],
    read: SubscriptPathReadContext,
    state: &mut super::SequenceState,
) -> Result<RootedValues, RuntimeError> {
    let mut argument_roots = Vec::new();
    let (root, path, _roots) = if let [crate::bytecode::BytecodeSubscriptStep::DottedInvoke {
        member,
        arguments,
        semantic_fallback,
    }] = steps
    {
        use crate::bytecode::BytecodeDottedInvokeArgument as Argument;
        use runmat_runtime::object::indexing::{
            ObjectIndexComponent, ObjectSubscript, ObjectSubscriptPath,
        };
        let mut groups = Vec::with_capacity(arguments.len());
        for argument in arguments.iter().rev() {
            match argument {
                Argument::Colon => groups.push(vec![ObjectIndexComponent::Colon]),
                Argument::Call(spec) => {
                    let values = crate::call::shared::build_expanded_args_from_specs(
                        stack,
                        std::slice::from_ref(spec),
                        state,
                        runtime,
                    )
                    .await?;
                    values.with_unbounded_values(|values| {
                        argument_roots.push(crate::call::root_values(&values)?);
                        groups.push(
                            values
                                .into_iter()
                                .map(ObjectIndexComponent::Value)
                                .collect(),
                        );
                        Ok::<_, RuntimeError>(())
                    })??;
                }
            }
        }
        groups.reverse();
        let root = stack.pop().ok_or_else(|| {
            crate::interpreter::errors::mex("StackUnderflow", "dotted invocation has no receiver")
        })?;
        let components = groups.into_iter().flatten().collect();
        let path = ObjectSubscriptPath::new(
            ObjectSubscript::dotted_invoke_components_with_fallback(
                member.clone(),
                components,
                semantic_fallback.clone(),
            )
            .into(),
        )?;
        let roots = crate::call::root_values(std::iter::once(&root))?;
        (root, Some(path), roots)
    } else {
        let operand_count = steps
            .iter()
            .map(crate::bytecode::BytecodeSubscriptStep::operand_count)
            .sum::<usize>();
        let start = stack.len().checked_sub(operand_count + 1).ok_or_else(|| {
            crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow")
        })?;
        let operands = stack
            .try_extract_output(start)?
            .try_into_owned()
            .map_err(|_| {
                crate::interpreter::errors::mex(
                    "SharedOutputStorage",
                    "subscript operands unexpectedly have shared storage",
                )
            })?;
        let roots = crate::call::root_values(operands.as_slice())?;
        let (root, path) = materialize_subscript_path(operands, steps)?;
        (root, path, roots)
    };
    let path = path.ok_or_else(|| {
        crate::interpreter::errors::mex(
            "RunMat:InvalidObjectSubscriptPath",
            "subscript path is empty",
        )
    })?;
    let sequence_context = match read.selection {
        runmat_types::SequenceUse::SelectCurrentFunctionOutputs => {
            runmat_runtime::sequence::SequenceResolutionContext::current_function_outputs(
                read.invocation_arity
                    .function_counts()
                    .map_or(1, |(_, outputs)| outputs),
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
        sequence_use: read.selection,
        sequence_context,
        indexing_context: read.indexing,
        access: read.access,
    };
    let sequence =
        runmat_runtime::object::protocol::read_subscript_path_sequence_with_access_in_context(
            runtime, root, path, request,
        )
        .await?;
    sequence.with_owned_sequence(|sequence| {
        RootedValues::from_output_storage(sequence.resolve(read.selection, sequence_context)?)
            .map_err(|error| error.into_runtime_error())
    })
}

fn materialize_subscript_path(
    values: impl IntoIterator<Item = Value>,
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
            Step::DottedInvoke { .. } => {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "dotted invocation arguments must be materialized by the call argument service",
                ))
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
