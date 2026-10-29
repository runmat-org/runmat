use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValues;
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

mod capture;
mod read;
mod subscript;
mod suspend;
mod write;

pub(super) fn write_sequence_values(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    register: &mut super::SequenceState,
    values: impl ExactSizeIterator<Item = Value>,
) -> Result<(), RuntimeError> {
    if register.assignment.is_some() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "sequence register was overwritten before consumption",
        ));
    }
    let start = stack.len();
    let len = values.len();
    stack.try_extend(values)?;
    register.assignment = Some(super::SequenceRegister { start, len });
    Ok(())
}

pub(super) fn write_rooted_sequence_register(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    register: &mut super::SequenceState,
    values: RootedValues,
) -> Result<(), RuntimeError> {
    write_sequence_values(stack, register, values.try_into_values()?)
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

pub(super) use suspend::{dispatch_object, Request};

pub(super) fn frame_access(
    bytecode: &crate::bytecode::Bytecode,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::object::protocol::ObjectAccessContext, RuntimeError> {
    let _scope = runtime.enter();
    if let (Some(owner), Some(function)) = (
        bytecode.active_class_method_owner.as_ref(),
        bytecode.active_function,
    ) {
        if let Some(access) = runtime.selected_method_frame_access(owner, function)? {
            return Ok(access);
        }
    }
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

/// A borrowed view of the existing bytecode selector/argument descriptors.
/// It carries no execution authority and never clones MemberName strings.
#[derive(Clone, Copy)]
pub(super) enum PreparedSelectorSource<'a> {
    Colon,
    Value,
    Expansion(&'a runmat_runtime::call::arguments::ArgumentSpec),
}

pub(super) fn selector_source(
    selector: &crate::bytecode::BytecodeSubscriptSelector,
) -> PreparedSelectorSource<'_> {
    match selector {
        crate::bytecode::BytecodeSubscriptSelector::Colon => PreparedSelectorSource::Colon,
        crate::bytecode::BytecodeSubscriptSelector::Value => PreparedSelectorSource::Value,
        crate::bytecode::BytecodeSubscriptSelector::Expansion(spec) => {
            PreparedSelectorSource::Expansion(spec)
        }
    }
}

impl<'a> PreparedSelectorSource<'a> {
    pub(super) fn argument(self) -> Option<&'a runmat_runtime::call::arguments::ArgumentSpec> {
        match self {
            Self::Colon => None,
            Self::Value => Some(&runmat_runtime::call::arguments::ArgumentSpec::Single),
            Self::Expansion(spec) => Some(spec),
        }
    }
}

pub(super) async fn materialize_ordered_selector_arguments<'a>(
    runtime: &runmat_runtime::context::RuntimeContext,
    arguments: &mut impl Iterator<Item = runmat_runtime::call::arguments::MaterializedArgument>,
    selectors: impl Iterator<Item = PreparedSelectorSource<'a>> + Clone,
    argument_roots: &mut crate::runtime::workspace::FrameVec<
        runmat_runtime::execution::RootedValueLease,
    >,
    state: &super::SequenceState,
) -> Result<
    runmat_value::ExactArrayStorage<runmat_runtime::object::indexing::ObjectIndexComponent>,
    RuntimeError,
> {
    use runmat_runtime::call::arguments::MaterializedArgument;
    use runmat_runtime::object::indexing::ObjectIndexComponent;
    use PreparedSelectorSource as Selector;
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?;
    state.validate_capture_context(runtime)?;
    // The fixed source descriptor bounds this control ledger before getters.
    // Dynamic expansion outputs retain their canonical physical owners until
    // the complete width is known and the final component storage is admitted.
    let mut segments = state.frame_vector(selectors.clone().count())?;
    let mut width = 0usize;
    for selector in selectors {
        let segment = if matches!(selector, Selector::Colon) {
            width = width.checked_add(1).ok_or_else(selector_layout_error)?;
            None
        } else {
            let argument = arguments.next().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "prepared selector is missing",
                )
            })?;
            let argument = match (selector, argument) {
                (Selector::Value, MaterializedArgument::Single(value))
                | (Selector::Expansion(_), MaterializedArgument::Single(value)) => {
                    MaterializedArgument::Single(value)
                }
                (Selector::Expansion(_), MaterializedArgument::Expansion(source)) => {
                    let sequence =
                        runmat_runtime::call::arguments::materialize_expansion(runtime, source)
                            .await?;
                    push_selector_root(argument_roots, sequence.root_lease())?;
                    MaterializedArgument::Sequence(sequence)
                }
                (Selector::Expansion(_), MaterializedArgument::Sequence(sequence)) => {
                    push_selector_root(argument_roots, sequence.root_lease())?;
                    MaterializedArgument::Sequence(sequence)
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "RunMat:SubscriptPathState",
                        "prepared selector kind does not match its argument",
                    ))
                }
            };
            let count = match &argument {
                MaterializedArgument::Single(_) => 1,
                MaterializedArgument::Sequence(sequence) => sequence.value().len(),
                MaterializedArgument::Expansion(_) => unreachable!("source was materialized once"),
            };
            width = width.checked_add(count).ok_or_else(selector_layout_error)?;
            Some(argument)
        };
        segments.push(segment);
        cooperation
            .checkpoint(1)
            .await
            .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?;
    }
    let mut components =
        runmat_runtime::object::indexing::prepare_components_in_context(runtime, width)?;
    for segment in segments {
        match segment {
            None => components
                .try_push(ObjectIndexComponent::Colon)
                .map_err(|_| selector_layout_error())?,
            Some(MaterializedArgument::Single(value)) => components
                .try_push(ObjectIndexComponent::Value(value))
                .map_err(|_| selector_layout_error())?,
            Some(MaterializedArgument::Sequence(sequence)) => {
                let values = sequence.try_into_values()?;
                for value in values {
                    components
                        .try_push(ObjectIndexComponent::Value(value))
                        .map_err(|_| selector_layout_error())?;
                    cooperation
                        .checkpoint(1)
                        .await
                        .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?;
                }
            }
            Some(MaterializedArgument::Expansion(_)) => unreachable!("materialized segment"),
        }
    }
    let components = components
        .try_finish()
        .map_err(|_| selector_layout_error())?;
    let roots = runmat_runtime::execution::RootedValueLease::for_values_iterator_in_context(
        components
            .as_slice()
            .iter()
            .filter_map(|component| match component {
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
                ObjectIndexComponent::Value(value) => Some(value),
            }),
        runtime,
    )?;
    push_selector_root(argument_roots, roots)?;
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?;
    Ok(components)
}

pub(super) fn push_selector_root(
    roots: &mut crate::runtime::workspace::FrameVec<runmat_runtime::execution::RootedValueLease>,
    value: runmat_runtime::execution::RootedValueLease,
) -> Result<(), RuntimeError> {
    if roots.len() == roots.capacity() {
        return Err(selector_layout_error());
    }
    roots.push(value);
    Ok(())
}

fn selector_layout_error() -> RuntimeError {
    runmat_runtime::execution::ExecutionServiceError::Infrastructure(
        "prepared selector control geometry disagrees with its admitted source descriptor".into(),
    )
    .into_runtime_error()
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
    use crate::bytecode::{
        BytecodeDottedInvokeArgument as Argument, BytecodeSubscriptStep as Step,
    };
    use runmat_runtime::call::arguments::{ArgumentSpec, MaterializedArgument};
    use runmat_runtime::object::indexing::{
        ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
    };
    state.validate_capture_context(runtime)?;
    let root_capacity = steps.iter().try_fold(2usize, |total, step| {
        let count = match step {
            Step::Parentheses { selectors } | Step::Braces { selectors } => selectors.len(),
            Step::DottedInvoke { arguments, .. } => arguments.len(),
            _ => 0,
        };
        total
            .checked_add(count)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(selector_layout_error)
    })?;
    let mut argument_roots = state.frame_vector(root_capacity)?;
    let (root, path, _roots) = if let [Step::DottedInvoke { member, arguments, semantic_fallback }] = steps {
        let inputs = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(&[stack.as_slice()], runtime)?;
        let selectors = arguments.iter().map(|argument| match argument {
            Argument::Colon => PreparedSelectorSource::Colon,
            Argument::Call(ArgumentSpec::Single) => PreparedSelectorSource::Value,
            Argument::Call(spec) => PreparedSelectorSource::Expansion(spec),
        });
        let arguments = crate::call::shared::take_materialized_args_from_specs(
            stack, selectors.clone().filter_map(PreparedSelectorSource::argument), state, runtime)?;
        let root = stack.pop().ok_or_else(|| crate::interpreter::errors::mex(
            "StackUnderflow", "dotted invocation has no receiver"))?;
        let mut arguments = arguments.into_iter();
        let mut path = runmat_runtime::object::indexing::prepare_subscripts_in_context(runtime, 2)?;
        let components = materialize_ordered_selector_arguments(
            runtime, &mut arguments, selectors, &mut argument_roots, state).await?;
        if arguments.next().is_some() { return Err(selector_layout_error()); }
        let member = runmat_runtime::call::argument_text::copy_str(&member.0, Some(runtime))?;
        for step in ObjectSubscript::dotted_invoke_retained_with_fallback(
            member, components, semantic_fallback.clone()) {
            path.try_push(step).map_err(|_| selector_layout_error())?;
        }
        let path = ObjectSubscriptPath::from_storage(path.try_finish().map_err(|_| selector_layout_error())?)?;
        (root, Some(path), inputs)
    } else if steps.iter().any(|step| matches!(step, Step::Parentheses { selectors }
        if selectors.iter().any(|selector| matches!(selector, crate::bytecode::BytecodeSubscriptSelector::Expansion(_))))) {
        if steps.iter().any(|step| matches!(step, Step::DottedInvoke { .. })) {
            return Err(crate::interpreter::errors::mex("RunMat:SubscriptPathState",
                "dotted invocation requires its own prepared segment"));
        }
        let inputs = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(&[stack.as_slice()], runtime)?;
        let specs = steps.iter().flat_map(|step| {
            let (dynamic, selectors): (bool, &[crate::bytecode::BytecodeSubscriptSelector]) = match step {
                Step::DynamicMember => (true, &[]),
                Step::Parentheses { selectors } | Step::Braces { selectors } => (false, selectors),
                _ => (false, &[]),
            };
            std::iter::once(&ArgumentSpec::Single).take(usize::from(dynamic))
                .chain(selectors.iter().filter_map(|selector| selector_source(selector).argument()))
        });
        let arguments = crate::call::shared::take_materialized_args_from_specs(stack, specs, state, runtime)?;
        let root = stack.pop().ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "subscript path has no receiver"))?;
        let mut arguments = arguments.into_iter();
        let mut path = runmat_runtime::object::indexing::prepare_subscripts_in_context(runtime, steps.len())?;
        for step in steps {
            let step = match step {
                Step::Member(member) => ObjectSubscript::member_retained(
                    runmat_runtime::call::argument_text::copy_str(&member.0, Some(runtime))?),
                Step::DynamicMember => {
                    let Some(MaterializedArgument::Single(value)) = arguments.next() else { return Err(selector_layout_error()); };
                    ObjectSubscript::member_retained(runmat_runtime::call::argument_text::convert(&value, Some(runtime))?)
                }
                Step::Parentheses { selectors } | Step::Braces { selectors } => {
                    let components = materialize_ordered_selector_arguments(
                        runtime, &mut arguments, selectors.iter().map(selector_source), &mut argument_roots, state).await?;
                    let selector = ObjectIndexSelector::IndexValues { components };
                    if matches!(step, Step::Parentheses { .. }) { ObjectSubscript::parentheses(selector) }
                    else { ObjectSubscript::braces(selector) }
                }
                Step::DottedInvoke { .. } => unreachable!("separate dotted segment checked"),
            };
            path.try_push(step).map_err(|_| selector_layout_error())?;
        }
        if arguments.next().is_some() { return Err(selector_layout_error()); }
        let path = ObjectSubscriptPath::from_storage(path.try_finish().map_err(|_| selector_layout_error())?)?;
        (root, Some(path), inputs)
    } else {
        let operand_count = steps.iter().try_fold(0usize, |total, step|
            total.checked_add(step.operand_count()).ok_or_else(selector_layout_error))?;
        let start = stack.len().checked_sub(operand_count.checked_add(1).ok_or_else(selector_layout_error)?)
            .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow"))?;
        let operands = stack.try_extract_output(start)?.try_into_owned().map_err(|_| selector_layout_error())?;
        let roots = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(&[operands.as_slice()], runtime)?;
        let (root, path) = materialize_subscript_path(runtime, operands, steps)?;
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
        RootedValues::from_output_storage_in_context(
            sequence.resolve(read.selection, sequence_context)?,
            runtime,
        )
    })
}

fn materialize_subscript_path(
    runtime: &RuntimeContext,
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
    let mut runtime_steps =
        runmat_runtime::object::indexing::prepare_subscripts_in_context(runtime, steps.len())?;
    for step in steps {
        match step {
            Step::Member(member) => runtime_steps
                .try_push(ObjectSubscript::member_retained(
                    runmat_runtime::call::argument_text::copy_str(&member.0, Some(runtime))?,
                ))
                .map_err(|_| selector_layout_error())?,
            Step::DynamicMember => {
                let value = values.next().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "dynamic member is missing")
                })?;
                runtime_steps
                    .try_push(ObjectSubscript::member_retained(
                        runmat_runtime::call::argument_text::convert(&value, Some(runtime))?,
                    ))
                    .map_err(|_| selector_layout_error())?;
            }
            Step::Parentheses { selectors } | Step::Braces { selectors } => {
                let components = materialize_selector_components(runtime, &mut values, selectors)?;
                let selector = ObjectIndexSelector::IndexValues { components };
                runtime_steps
                    .try_push(if matches!(step, Step::Parentheses { .. }) {
                        ObjectSubscript::parentheses(selector)
                    } else {
                        ObjectSubscript::braces(selector)
                    })
                    .map_err(|_| selector_layout_error())?;
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
    let path = if steps.is_empty() {
        None
    } else {
        Some(ObjectSubscriptPath::from_storage(
            runtime_steps
                .try_finish()
                .map_err(|_| selector_layout_error())?,
        )?)
    };
    Ok((root, path))
}

fn materialize_selector_components(
    runtime: &RuntimeContext,
    values: &mut impl Iterator<Item = Value>,
    selectors: &[crate::bytecode::BytecodeSubscriptSelector],
) -> Result<
    runmat_value::ExactArrayStorage<runmat_runtime::object::indexing::ObjectIndexComponent>,
    RuntimeError,
> {
    use crate::bytecode::BytecodeSubscriptSelector as Selector;
    use runmat_runtime::object::indexing::ObjectIndexComponent;
    let mut components =
        runmat_runtime::object::indexing::prepare_components_in_context(runtime, selectors.len())?;
    for selector in selectors {
        let component = match selector {
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
            Selector::Expansion(_) => Err(crate::interpreter::errors::mex(
                "RunMat:SubscriptPathState",
                "expanded selector requires prepared parentheses materialization",
            )),
        }?;
        components
            .try_push(component)
            .map_err(|_| selector_layout_error())?;
    }
    components.try_finish().map_err(|_| selector_layout_error())
}
