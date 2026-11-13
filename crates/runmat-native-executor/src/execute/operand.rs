use runmat_mir::{MirConstant, MirOperand, MirRvalue};
use runmat_runtime::execution::RootedValueLease;
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

fn sequence_resolution_context(
    sequence_use: runmat_types::SequenceUse,
    count: usize,
) -> runmat_runtime::sequence::SequenceResolutionContext {
    match sequence_use {
        runmat_types::SequenceUse::SelectDestinationCardinality => {
            runmat_runtime::sequence::SequenceResolutionContext::destination_cardinality(count)
        }
        runmat_types::SequenceUse::SelectCurrentFunctionOutputs => {
            runmat_runtime::sequence::SequenceResolutionContext::current_function_outputs(count)
        }
        _ => Default::default(),
    }
}

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "operand/sequence_read_tests.rs"]
mod sequence_read_tests;

pub(super) fn evaluate_rvalue(
    state: &mut HostState,
    value: &MirRvalue,
    requested_outputs: usize,
    _output_local: Option<runmat_native_codegen::NativeLocalId>,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    match value {
        MirRvalue::Use(operand) => evaluate_operand(state, operand).map(|value| vec![value]),
        MirRvalue::Unary(operator, operand) => {
            super::operator::evaluate(state, *operator, &[operand]).map(|value| vec![value])
        }
        MirRvalue::Binary(left, operator, right) => {
            super::operator::evaluate(state, *operator, &[left, right]).map(|value| vec![value])
        }
        MirRvalue::ShortCircuit {
            left,
            op,
            right_temps,
            right,
        } => super::short_circuit::evaluate(state, left, *op, right_temps, right),
        MirRvalue::Range { start, step, end } => {
            let mut arguments = vec![materialize_operand(state, start)?];
            if let Some(step) = step {
                arguments.push(materialize_operand(state, step)?);
            }
            arguments.push(materialize_operand(state, end)?);
            super::call::builtin(state, "colon", arguments, 1).and_then(|values| {
                super::call_suspension::own_output_storage(values)?
                    .into_iter()
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>()
            })
        }
        MirRvalue::Call(call) => {
            super::call::evaluate(state, call, requested_outputs).and_then(|values| {
                super::call_suspension::own_output_storage(values)?
                    .into_iter()
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>()
            })
        }
        MirRvalue::Aggregate {
            kind,
            row_lengths,
            elements,
        } => super::aggregate::evaluate(state, kind, row_lengths, elements)
            .map(|reference| vec![reference]),
        MirRvalue::StructLiteral { fields } => {
            super::aggregate::structure(state, fields).and_then(|rooted| {
                rooted.with_owned(|value| {
                    Ok(vec![state.arena.insert(value, Some(&state.runtime))?])
                })
            })
        }
        MirRvalue::ObjectLiteral { class_name, fields } => {
            super::aggregate::object(state, class_name, fields).and_then(|rooted| {
                rooted.with_owned(|value| {
                    Ok(vec![state.arena.insert(value, Some(&state.runtime))?])
                })
            })
        }
        MirRvalue::Index { base, indexing } => {
            super::indexing::read(state, base, indexing, requested_outputs).and_then(|values| {
                super::call_suspension::own_output_storage(values)?
                    .into_iter()
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>()
            })
        }
        MirRvalue::SubscriptChain(chain) => {
            super::subscript_path::read(state, chain, requested_outputs).and_then(|values| {
                values
                    .into_iter()
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>()
            })
        }
        MirRvalue::Member {
            base,
            member,
            sequence_use,
        } => {
            let stage = super::call_suspension::SequenceReadStage::Member {
                selection: *sequence_use,
                requested_outputs,
            };
            let values = if let Some(values) =
                super::call_suspension::take_completed_sequence_read(state, stage)?
            {
                values
            } else {
                let base = materialize_operand(state, base)?;
                let roots = RootedValueLease::for_values_iterator_in_context(
                    std::iter::once(&base),
                    &state.runtime,
                )?;
                let runtime = state.runtime.clone();
                let functions = std::sync::Arc::clone(&state.functions);
                let function = state.function.portable_id;
                let member =
                    runmat_runtime::call::argument_text::copy_str(&member.0, Some(&runtime))?;
                let access = super::subscript_path::frame_access(state)?;
                super::call_suspension::begin_sequence_read(state, stage, || async move {
                    let caller = functions
                        .iter()
                        .find(|candidate| candidate.portable_id == function)
                        .ok_or_else(|| {
                            NativeExecutorError::Host(
                                "member read lost canonical caller metadata".into(),
                            )
                        })?;
                    let sequence =
                        runmat_runtime::object::resolve::read_member_sequence_text_with_access(
                            Some(&runtime),
                            base,
                            member,
                            false,
                            Some(&caller.name),
                            &access,
                        )
                        .await?;
                    drop(roots);
                    Ok(sequence)
                })?
            };
            if *sequence_use == runmat_types::SequenceUse::RequireSingle {
                return values.try_into_single()?.with_owned(|value| {
                    Ok(vec![state.arena.insert(value, Some(&state.runtime))?])
                });
            }
            if *sequence_use == runmat_types::SequenceUse::ExpandAll {
                return values
                    .try_into_values()?
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>();
            }
            values
                .resolve_values(
                    *sequence_use,
                    sequence_resolution_context(*sequence_use, requested_outputs),
                )?
                .map(|value| state.arena.insert(value, Some(&state.runtime)))
                .collect::<NativeExecutorResult<_>>()
        }
        MirRvalue::DynamicMember {
            base,
            member,
            sequence_use,
        } => {
            let stage = super::call_suspension::SequenceReadStage::DynamicMember {
                selection: *sequence_use,
                requested_outputs,
            };
            let values = if let Some(values) =
                super::call_suspension::take_completed_sequence_read(state, stage)?
            {
                values
            } else {
                let base = materialize_operand(state, base)?;
                let member = materialize_operand(state, member)?;
                let roots = RootedValueLease::for_values_iterator_in_context(
                    [&base, &member].into_iter(),
                    &state.runtime,
                )?;
                let member =
                    runmat_runtime::call::argument_text::convert(&member, Some(&state.runtime))?;
                let runtime = state.runtime.clone();
                let access = super::subscript_path::frame_access(state)?;
                let functions = std::sync::Arc::clone(&state.functions);
                let function = state.function.portable_id;
                super::call_suspension::begin_sequence_read(state, stage, || async move {
                    let caller = functions
                        .iter()
                        .find(|candidate| candidate.portable_id == function)
                        .ok_or_else(|| {
                            NativeExecutorError::Host(
                                "dynamic member read lost canonical caller metadata".into(),
                            )
                        })?;
                    let sequence =
                        runmat_runtime::object::resolve::read_member_sequence_text_with_access(
                            Some(&runtime),
                            base,
                            member,
                            false,
                            Some(&caller.name),
                            &access,
                        )
                        .await?;
                    drop(roots);
                    Ok(sequence)
                })?
            };
            if *sequence_use == runmat_types::SequenceUse::RequireSingle {
                return values.try_into_single()?.with_owned(|value| {
                    Ok(vec![state.arena.insert(value, Some(&state.runtime))?])
                });
            }
            if *sequence_use == runmat_types::SequenceUse::ExpandAll {
                return values
                    .try_into_values()?
                    .map(|value| state.arena.insert(value, Some(&state.runtime)))
                    .collect::<NativeExecutorResult<_>>();
            }
            values
                .resolve_values(
                    *sequence_use,
                    sequence_resolution_context(*sequence_use, requested_outputs),
                )?
                .map(|value| state.arena.insert(value, Some(&state.runtime)))
                .collect::<NativeExecutorResult<_>>()
        }
        MirRvalue::WorkspaceFirstStaticProperty {
            workspace_name,
            class_name,
            property,
        } => {
            let value = runmat_runtime::workspace::lookup(&workspace_name.0)
                .or_else(|| {
                    state
                        .function
                        .locals
                        .iter()
                        .find(|local| local.name.as_deref() == Some(workspace_name.0.as_str()))
                        .and_then(|local| state.locals.get(local.id.0 as usize))
                        .copied()
                        .filter(|value| !value.is_null())
                        .and_then(|value| state.arena.get(value).ok().cloned())
                })
                .map(Ok)
                .unwrap_or_else(|| {
                    runmat_runtime::object::resolve::load_static_member(
                        class_name,
                        &property.0,
                        Some(&state.function.name),
                    )
                    .map_err(NativeExecutorError::from)
                })?;
            Ok(vec![state.arena.insert(value, Some(&state.runtime))?])
        }
        MirRvalue::MetaClass(name) => Ok(vec![state.arena.insert(
            Value::String(
                name.0
                    .iter()
                    .map(|segment| segment.0.as_str())
                    .collect::<Vec<_>>()
                    .join(".")
                    .into(),
            ),
            Some(&state.runtime),
        )?]),
        MirRvalue::Colon => Ok(vec![state
            .arena
            .insert(Value::Num(0.0), Some(&state.runtime))?]),
        MirRvalue::End => {
            if let Some((prepared, component, component_count)) =
                state.subscript_end_receivers.last()
            {
                let prepared = prepared.clone();
                let component = *component;
                let component_count = *component_count;
                let value = super::subscript_path::resolve_end(
                    state,
                    prepared,
                    component,
                    component_count,
                )?;
                return Ok(vec![state.arena.insert(value, Some(&state.runtime))?]);
            }
            if state.contextual_index_extents.is_empty()
                && super::mutation::has_fixed_end_context(state)
            {
                return super::mutation::resolve_fixed_end(state).map(|value| vec![value]);
            }
            let extent = state
                .contextual_index_extents
                .last()
                .copied()
                .ok_or_else(|| {
                    NativeExecutorError::Host(
                        "verified Native IR evaluated end without an indexing context".into(),
                    )
                })?;
            Ok(vec![state
                .arena
                .insert(Value::Num(extent as f64), Some(&state.runtime))?])
        }
        MirRvalue::Future {
            function,
            args,
            requested_outputs,
            ..
        }
        | MirRvalue::ExternalFuture {
            function,
            args,
            requested_outputs,
            ..
        } => {
            let arguments = super::call::materialize_arguments(state, args)?;
            runmat_runtime::execution::validate_spawn_captures(&arguments)?;
            let identity = match value {
                MirRvalue::ExternalFuture { display_name, .. } => {
                    runmat_types::CallableIdentity::ExternalFunction {
                        function: *function,
                        display_name: display_name.clone(),
                    }
                }
                _ => runmat_types::CallableIdentity::BoundFunction(*function),
            };
            let selected_function = state
                .runtime
                .selected_graph_function_binding(state.function.id, &identity);
            let spawn_safety = if let Some(selected) = &selected_function {
                selected.spawn_safety.clone()
            } else {
                state
                    .functions
                    .iter()
                    .find(|candidate| candidate.id == *function)
                    .map(|function| function.spawn_safety.clone())
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "async callable has no exact Native IR spawn contract".into(),
                        )
                    })?
            };
            let capabilities = match &selected_function {
                Some(selected) => selected.capabilities.clone(),
                None if matches!(
                    identity,
                    runmat_types::CallableIdentity::ExternalFunction { .. }
                ) =>
                {
                    return Err(NativeExecutorError::Host(
                        "native external future has no exact selected function contract".into(),
                    ));
                }
                None => native_function_capabilities(state, *function)?,
            };
            let mut invocation = runmat_runtime::execution::DeferredInvocation::callable(
                runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    identity.clone(),
                    arguments,
                    requested_outputs.executor_carrier_count().ok_or_else(|| {
                        NativeExecutorError::Host(
                            "future output cardinality cannot be derived from an assignment destination"
                                .into(),
                        )
                    })?,
                    runmat_types::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
                ),
            );
            if let Some(selected) = selected_function {
                let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) =
                    &mut invocation
                else {
                    unreachable!("native future admission created a callable invocation")
                };
                descriptor.target =
                    runmat_runtime::call::descriptor::CallableTarget::Bound(selected.binding);
            }
            let program = if !state.runtime.execution().requires_program_capture() {
                runmat_runtime::execution::DeferredProgram::Local
            } else if let Some(selected) = state.runtime.selected_graph_program() {
                if !matches!(&invocation,
                    runmat_runtime::execution::DeferredInvocation::Callable(descriptor)
                    if matches!(&descriptor.target, runmat_runtime::call::descriptor::CallableTarget::Bound(_))
                ) {
                    return Err(NativeExecutorError::Host(
                        "native async callable has no exact selected definition and source graph"
                            .into(),
                    ));
                }
                runmat_runtime::execution::DeferredProgram::SelectedGraph(selected.clone())
            } else {
                if runmat_runtime::execution::value_codec::deferred_invocation_requires_graph(
                    &invocation,
                ) {
                    return Err(NativeExecutorError::Host(
                        "native async lexical capture requires selected V3 graph authority".into(),
                    ));
                }
                let bytes = state.program_capture.clone().ok_or_else(|| {
                    NativeExecutorError::Host(
                        "native async execution is missing its exact program".into(),
                    )
                })?;
                runmat_runtime::execution::DeferredProgram::InterpreterV2(std::sync::Arc::from(
                    bytes,
                ))
            };
            let program_revision = program
                .program_revision(state.runtime.program_revision())
                .cloned();
            let future = state
                .runtime
                .execution()
                .create_future(runmat_runtime::execution::DeferredCall {
                    spawn_safety,
                    invocation,
                    retry: runmat_execution::RetryPolicy::Never,
                    environment: runmat_runtime::execution::capture_deferred_environment(
                        &state.runtime,
                    )?,
                    console: state
                        .runtime
                        .service_ports()
                        .host()
                        .and_then(|host| host.console_limits()),
                    program_revision,
                    capabilities,
                    program,
                })
                .map_err(execution_service_error)?;
            Ok(vec![state.arena.insert(
                Value::Future(future.into()),
                Some(&state.runtime),
            )?])
        }
        MirRvalue::Spawn(operand) => {
            let value = materialize_operand(state, operand)?;
            let Value::Future(future) = value else {
                return Err(runmat_runtime::runtime_error::semantic_error(
                    "SpawnOperandInvalid",
                    "spawn expects a lazy future produced by an async call",
                )
                .into());
            };
            let task = state
                .runtime
                .execution()
                .spawn_in_context(&future, None, &state.runtime)
                .map_err(execution_service_error)?;
            Ok(vec![state
                .arena
                .insert(Value::Task(task.into()), Some(&state.runtime))?])
        }
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => Err(NativeExecutorError::Host(
            "predeclared distributed capability rejection reached native execution".into(),
        )),
    }
}

fn native_function_capabilities(
    state: &super::state::HostState,
    function: runmat_types::FunctionId,
) -> NativeExecutorResult<runmat_types::CapabilitySet> {
    state
        .functions
        .iter()
        .find(|candidate| candidate.id == function)
        .map(|function| function.capabilities.clone())
        .ok_or_else(|| {
            NativeExecutorError::Host(
                "async callable is absent from the exact Native IR function set".into(),
            )
        })
}

fn execution_service_error(
    error: runmat_runtime::execution::ExecutionServiceError,
) -> NativeExecutorError {
    error.into_runtime_error().into()
}

pub(super) fn materialize_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> NativeExecutorResult<Value> {
    let reference = evaluate_operand(state, operand)?;
    state.arena.get(reference).cloned()
}

pub(super) fn materialize_retained_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> NativeExecutorResult<(NativeValueRef, Value)> {
    let reference = if let Some(reference) = state.take_contextual_input() {
        reference
    } else {
        let reference = evaluate_operand(state, operand)?;
        state.retain_contextual_input(reference)?
    };
    Ok((reference, state.arena.get(reference)?.clone()))
}

pub(super) fn evaluate_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> NativeExecutorResult<NativeValueRef> {
    match operand {
        MirOperand::Local(local) => {
            let value = state.local_reference(local.0)?;
            if value.is_null()
                && u32::try_from(local.0)
                    .ok()
                    .map(runmat_native_codegen::NativeLocalId)
                    .is_some_and(|local| state.function.abi.fixed_inputs.contains(&local))
            {
                return Err(NativeExecutorError::from(
                    runmat_runtime::runtime_error::semantic_error(
                        "NotEnoughInputs",
                        "Not enough input arguments.",
                    ),
                ));
            }
            Ok(value)
        }
        MirOperand::Constant(constant) => {
            let value = match constant {
                MirConstant::Number(text) => Value::Num(text.parse::<f64>().map_err(|error| {
                    NativeExecutorError::Host(format!(
                        "invalid MIR numeric constant {text:?}: {error}"
                    ))
                })?),
                MirConstant::ImaginaryNumber(text) => Value::Complex(
                    0.0,
                    text.parse::<f64>().map_err(|error| {
                        NativeExecutorError::Host(format!(
                            "invalid MIR imaginary numeric constant {text:?}: {error}"
                        ))
                    })?,
                ),
                MirConstant::IntegerLiteral(value) => {
                    Value::Int(runmat_value::IntValue::from(value))
                }
                MirConstant::String(text) if text.is_character_row() => {
                    Value::CharArray(runmat_value::CharArray::new_row(&text.runtime_text()))
                }
                MirConstant::String(text) => Value::String(text.runtime_text().into()),
                MirConstant::Symbol(symbol) => {
                    runmat_runtime::builtin::builtin_constant_value(&symbol.0).ok_or_else(|| {
                        NativeExecutorError::Host(format!("unknown MIR constant {}", symbol.0))
                    })?
                }
                MirConstant::Bool(value) => Value::Bool(*value),
                MirConstant::EmptyArray => Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                        .map_err(NativeExecutorError::Host)?,
                ),
            };
            state.arena.insert(value, Some(&state.runtime))
        }
        MirOperand::FunctionHandle(identity) => {
            if let runmat_types::CallableIdentity::BoundFunction(function)
            | runmat_types::CallableIdentity::AnonymousFunction(function) = identity
            {
                let function = *function;
                if let Some(target) = state.program_function(function) {
                    let name = target.name.clone();
                    let source_role = target.source_role;
                    let carries_environment = source_role
                        == runmat_types::FunctionSourceRole::Nested
                        || (source_role == runmat_types::FunctionSourceRole::Anonymous
                            && target.lexical_bindings.iter().any(|binding| {
                                binding.kind
                                    == runmat_native_codegen::NativeLexicalBindingKind::Transit
                            }));
                    let captures = state.callable_capture_values(function)?;
                    let value = if captures.is_empty()
                        && source_role != runmat_types::FunctionSourceRole::Nested
                    {
                        Value::bound_function_handle(
                            name,
                            function.0,
                            runmat_runtime::user_functions::bound_function_authority_for(
                                function.0,
                            ),
                            source_role,
                            runmat_runtime::user_functions::source_file_for(function.0),
                        )
                    } else {
                        runmat_runtime::call::closures::semantic_closure_value(
                            function,
                            name,
                            captures,
                            source_role,
                            carries_environment
                                .then(|| state.lexical_environment.clone())
                                .flatten(),
                        )
                    };
                    return state.arena.insert(value, Some(&state.runtime));
                }
            }
            identity
                .display_name()
                .map(Value::function_handle)
                .map(|value| state.arena.insert(value, Some(&state.runtime)))
                .ok_or_else(|| {
                    NativeExecutorError::Host(format!(
                        "function handle {identity:?} has no runtime identity"
                    ))
                })?
        }
    }
}
