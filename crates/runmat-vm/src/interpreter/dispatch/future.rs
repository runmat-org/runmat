//! Poll-shape owner for one VM instruction.
//!
//! Immediate instructions complete without allocation. A selected instruction
//! family owns the only boxed future when the operation can actually suspend.

use super::*;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

type DispatchResult = Result<Option<DispatchHandled>, RuntimeError>;
type SuspendedDispatchFuture<'a> = Pin<Box<dyn Future<Output = DispatchResult> + 'a>>;

pub(super) struct InstructionDispatchFuture<'a> {
    state: InstructionDispatchFutureState<'a>,
}

enum InstructionDispatchFutureState<'a> {
    Immediate(Option<ImmediateInstruction<'a>>),
    Ready(Option<DispatchResult>),
    Arithmetic(arithmetic::ArithmeticDispatchFuture<'a>),
    Suspended(SuspendedDispatchFuture<'a>),
}

struct ImmediateInstruction<'a> {
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
}

impl Future for InstructionDispatchFuture<'_> {
    type Output = DispatchResult;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        match &mut this.state {
            InstructionDispatchFutureState::Immediate(instruction) => {
                let instruction = instruction
                    .take()
                    .expect("completed instruction future was polled again");
                Poll::Ready(dispatch_immediate(
                    instruction.meta,
                    instruction.state,
                    instruction.hooks,
                ))
            }
            InstructionDispatchFutureState::Ready(result) => Poll::Ready(
                result
                    .take()
                    .expect("completed instruction future was polled again"),
            ),
            InstructionDispatchFutureState::Arithmetic(future) => {
                match future.as_mut().poll(context) {
                    Poll::Ready(Ok(())) => Poll::Ready(Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )))),
                    Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
                    Poll::Pending => Poll::Pending,
                }
            }
            InstructionDispatchFutureState::Suspended(future) => future.as_mut().poll(context),
        }
    }
}

impl<'a> InstructionDispatchFuture<'a> {
    fn immediate(
        meta: DispatchMeta<'a>,
        state: DispatchState<'a>,
        hooks: DispatchHooks<'a>,
    ) -> Self {
        Self {
            state: InstructionDispatchFutureState::Immediate(Some(ImmediateInstruction {
                meta,
                state,
                hooks,
            })),
        }
    }

    fn suspended(future: SuspendedDispatchFuture<'a>) -> Self {
        Self {
            state: InstructionDispatchFutureState::Suspended(future),
        }
    }

    fn ready(result: DispatchResult) -> Self {
        Self {
            state: InstructionDispatchFutureState::Ready(Some(result)),
        }
    }

    fn arithmetic(future: arithmetic::ArithmeticDispatchFuture<'a>) -> Self {
        Self {
            state: InstructionDispatchFutureState::Arithmetic(future),
        }
    }
}

pub(super) fn dispatch_instruction<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> InstructionDispatchFuture<'a> {
    use route::InstructionDispatchFamily;

    match route::family(meta.instr) {
        InstructionDispatchFamily::Immediate => {
            InstructionDispatchFuture::immediate(meta, state, hooks)
        }
        InstructionDispatchFamily::SemanticClosure => {
            let pending = matches!(meta.instr, Instr::CreateSemanticClosure(function, ..)
            if state.context.dynamic_eval_admission.as_ref().is_some_and(|port| {
                port.pending.as_ref().is_some_and(|plan| plan.function(*function).is_some())
            }));
            if pending {
                InstructionDispatchFuture::suspended(Box::pin(dispatch_semantic_closure(
                    meta, state,
                )))
            } else {
                InstructionDispatchFuture::immediate(meta, state, hooks)
            }
        }
        InstructionDispatchFamily::NameHandle => {
            InstructionDispatchFuture::suspended(Box::pin(dispatch_name_handle(meta, state)))
        }
        InstructionDispatchFamily::Parallel => {
            InstructionDispatchFuture::suspended(suspend_parallel(meta, state))
        }
        InstructionDispatchFamily::PreparedOutput => {
            InstructionDispatchFuture::suspended(suspend_prepared_output(meta, state, hooks))
        }
        InstructionDispatchFamily::Call => {
            InstructionDispatchFuture::suspended(dispatch_call(meta, state))
        }
        InstructionDispatchFamily::Indexing => {
            InstructionDispatchFuture::suspended(suspend_indexing(meta, state))
        }
        InstructionDispatchFamily::Object => {
            InstructionDispatchFuture::suspended(suspend_object(meta, state))
        }
        InstructionDispatchFamily::Arithmetic => match suspend_arithmetic(meta, state) {
            Ok(future) => InstructionDispatchFuture::arithmetic(future),
            Err(error) => InstructionDispatchFuture::ready(Err(error)),
        },
        InstructionDispatchFamily::Output => {
            InstructionDispatchFuture::suspended(suspend_output(meta, state))
        }
        InstructionDispatchFamily::Logical => {
            InstructionDispatchFuture::suspended(suspend_logical(meta, state))
        }
        InstructionDispatchFamily::Aggregate => {
            InstructionDispatchFuture::suspended(suspend_aggregate(meta, state))
        }
    }
}

async fn dispatch_semantic_closure(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
) -> DispatchResult {
    let Instr::CreateSemanticClosure(function, display_name, capture_count, source_role) =
        meta.instr
    else {
        unreachable!("semantic-closure route requires its creation opcode");
    };
    let inherits_shared_bindings = *source_role == runmat_types::FunctionSourceRole::Anonymous
        && meta.function_registry.get(*function).is_some_and(|target| {
            target.lexical_binding_slots.iter().any(|binding| {
                binding.kind == crate::bytecode::program::LexicalBindingKind::Transit
            })
        });
    let lexical_environment = (*source_role == runmat_types::FunctionSourceRole::Nested
        || inherits_shared_bindings)
        .then(|| state.context.lexical_environment.clone())
        .flatten();
    let port = state.context.dynamic_eval_admission.clone();
    let pending = port.as_ref().and_then(|port| {
        port.pending
            .as_ref()
            .filter(|plan| plan.function(*function).is_some())
            .cloned()
    });
    if let Some(plan) = pending {
        let port = port.expect("pending closure requires its admission port");
        let pending_function = plan
            .function(*function)
            .expect("pending closure lookup was checked above");
        if pending_function.abi.original_capture_count() != *capture_count {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure opcode differs from its original capture layout",
            ));
        }
        let runtime = state.context.runtime.clone();
        let compiled = crate::call::builtins::compile_pending_closure(
            &plan,
            pending_function,
            &runtime,
            meta.function_registry,
            *function,
            &port,
        )
        .await?;
        if crate::runtime::workspace::workspace_frame_identity() != Some(compiled.issuing_frame) {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure changed workspace frame after source selection",
            ));
        }
        let direct_captures = compiled.captures.len() == *capture_count
            && compiled.captures.iter().enumerate().all(|(index, capture)| {
                matches!(capture, crate::call::builtins::SelectedClosureCapture::Original(original) if *original == index)
            });
        let additional_captures = if direct_captures {
            None
        } else {
            if state.stack.len() < *capture_count {
                return Err(crate::interpreter::errors::mex(
                    "DynamicWorkspaceCompile",
                    "pending closure's original captures are absent from the stack",
                ));
            }
            let crate::runtime::workspace::WorkspaceBindingState {
                frame: live_frame,
                names: live_names,
                assigned,
                ..
            } = crate::runtime::workspace::workspace_binding_state().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "DynamicWorkspaceUnavailable",
                    "pending closure lost its issuing workspace frame",
                )
            })?;
            if live_frame != compiled.issuing_frame {
                return Err(crate::interpreter::errors::mex(
                    "DynamicWorkspaceCompile",
                    "pending closure changed workspace frame after source selection",
                ));
            }
            let mut values = Vec::with_capacity(compiled.captures.len());
            for capture in &compiled.captures {
                let crate::call::builtins::SelectedClosureCapture::AssignedWorkspace { name, slot } =
                    capture
                else {
                    continue;
                };
                if live_names.get(name) != Some(slot) || !assigned.contains(name) {
                    return Err(crate::interpreter::errors::mex(
                        "DynamicWorkspaceCompile",
                        "pending closure workspace capture changed after source selection",
                    ));
                }
                values.push(
                    crate::runtime::workspace::workspace_lookup(name).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "DynamicWorkspaceCompile",
                            "pending closure workspace capture is no longer assigned",
                        )
                    })?,
                );
            }
            Some(
                runmat_runtime::execution::RootedValues::new(values)
                    .map_err(|error| error.into_runtime_error())?,
            )
        };
        let admitted = port.host().admit_pending_closure(
            crate::interpreter::dynamic_admission::PendingClosureAdmissionRequest {
                runtime: &runtime,
                source: &plan.source,
                caller_source: plan.caller_source,
                product: compiled.product,
                compile_inherited: &plan.inherited,
                executing: meta.function_registry,
                local_function: pending_function.local_id,
                execution_function: *function,
                expected_issuing_owner: &pending_function.execution_owner,
                expected_source: pending_function.execution_source,
                siblings: &pending_function.siblings,
            },
        )?;
        if admitted.binding.function != function.0 {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure selected a different execution identity",
            ));
        }
        if let Some(captures) = additional_captures {
            let _stack_roots = runmat_runtime::execution::RootedValueLease::for_values(
                state.stack.as_slice()[state.stack.len() - *capture_count..].iter(),
            )
            .map_err(|error| error.into_runtime_error())?;
            let mut originals = Vec::with_capacity(*capture_count);
            for _ in 0..*capture_count {
                originals.push(crate::interpreter::stack::pop_value(state.stack)?);
            }
            originals.reverse();
            let mut originals = originals.into_iter().map(Some).collect::<Vec<_>>();
            captures.with_unbounded_values(|captures| -> Result<(), RuntimeError> {
                let mut added = captures.into_iter();
                let mut ordered = Vec::with_capacity(compiled.captures.len());
                for source in &compiled.captures {
                    let next = match source {
                        crate::call::builtins::SelectedClosureCapture::Original(index) => {
                            originals.get_mut(*index).and_then(Option::take)
                        }
                        crate::call::builtins::SelectedClosureCapture::AssignedWorkspace {
                            ..
                        } => added.next(),
                    }
                    .ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "DynamicWorkspaceCompile",
                            "pending closure capture plan lost a selected value",
                        )
                    })?;
                    ordered.push(next);
                }
                if originals.iter().any(Option::is_some) || added.next().is_some() {
                    return Err(crate::interpreter::errors::mex(
                        "DynamicWorkspaceCompile",
                        "pending closure capture plan left an issuing value unbound",
                    ));
                }
                crate::call::closures::create_selected_semantic_closure_with_captures(
                    state.stack,
                    admitted.binding,
                    display_name.clone(),
                    admitted.source_file,
                    ordered,
                    *source_role,
                    lexical_environment,
                )?;
                Ok(())
            })??;
        } else {
            crate::call::closures::create_selected_semantic_closure(
                state.stack,
                admitted.binding,
                display_name.clone(),
                admitted.source_file,
                *capture_count,
                *source_role,
                lexical_environment,
            )?;
        }
        drop(admitted.publication);
    } else {
        handle_create_semantic_closure(
            state.stack,
            *function,
            display_name.clone(),
            *capture_count,
            *source_role,
            lexical_environment,
        )?;
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn dispatch_name_handle(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    use runmat_runtime::user_functions::{
        DynamicFunctionLoadPhase, RuntimeNameCandidate, RuntimeNameFilter, RuntimeNameQuery,
        RuntimeNameRoute,
    };

    let (name, external) = match meta.instr {
        Instr::CreateFunctionHandle(name) => (name, false),
        Instr::CreateExternalFunctionHandle(name) => (name, true),
        _ => unreachable!("name-handle route requires a name-handle opcode"),
    };
    let runtime = state.context.runtime.clone();
    let selected = runtime
        .clone()
        .scope(async {
            let observed = runmat_runtime::user_functions::observe_runtime_name(
                name,
                RuntimeNameQuery {
                    route: RuntimeNameRoute::NamedFeval,
                    filter: RuntimeNameFilter::Any,
                    all: false,
                },
            )
            .await
            .map_err(|error| {
                runmat_runtime::build_runtime_error(error.to_string())
                    .with_identifier("RunMat:SelectedHandle:NameResolution")
                    .build()
            })?;
            let phase = match observed.selected() {
                Some(RuntimeNameCandidate::RegisteredConstructor(class)) => {
                    Some(DynamicFunctionLoadPhase::Handle {
                        class: Some(class.clone()),
                        selected_source: None,
                    })
                }
                Some(RuntimeNameCandidate::Source(file))
                | Some(RuntimeNameCandidate::ClassFile(file)) => {
                    Some(DynamicFunctionLoadPhase::Handle {
                        class: None,
                        selected_source: Some(file.clone()),
                    })
                }
                _ => None,
            };
            let Some(phase) = phase else {
                return Ok(None);
            };
            let selected = runmat_runtime::user_functions::try_load_and_call_dynamic_function(
                name.clone(),
                Vec::new(),
                1,
                phase,
                None,
            )
            .await;
            if runtime.search_path().map(|path| path.generation())
                != observed.captured_path_revision
                || runmat_runtime::class_registry::class_registry_generation()
                    != observed.captured_class_revision
            {
                return Err(runmat_runtime::build_runtime_error(
                    "name selection changed while creating a bound handle",
                )
                .with_identifier("RunMat:SelectedHandle:SelectionChanged")
                .build());
            }
            Ok(Some(selected))
        })
        .await?;
    if let Some(std::ops::ControlFlow::Break(result)) = selected {
        let rooted = result?;
        rooted.with_owned(|sequence| {
            if sequence.len() != 1
                || !matches!(
                    sequence.as_slice().first(),
                    Some(Value::BoundFunctionHandle(_))
                )
            {
                return Err(runmat_runtime::build_runtime_error(
                    "selected source did not return one exact bound handle",
                )
                .with_identifier("RunMat:SelectedHandle:InvalidOutput")
                .build());
            }
            let selected = sequence
                .into_output_storage()
                .try_into_first()
                .map_err(|_| {
                    runmat_runtime::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .expect("one checked selected handle");
            state.stack.try_push(selected)?;
            Ok(())
        })?;
    } else if external {
        state
            .stack
            .try_push(Value::external_function_handle(name.clone()))?;
    } else {
        state.stack.try_push(Value::function_handle(name.clone()))?;
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

// Keep construction of each concrete async family outside the exhaustive
// router. In debug builds, constructing all of these futures in one function
// makes that synchronous function reserve stack for the largest concrete
// future even though it returns only one boxed owner. Semantic Process calls
// construct a child instruction while every parent poll frame is live, so
// that transient maximum otherwise grows with ordinary call depth.
#[inline(never)]
fn suspend_parallel<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_parallel(meta, state))
}

#[inline(never)]
fn suspend_prepared_output<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_prepared_output(meta, state, hooks))
}

#[inline(never)]
fn suspend_indexing<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_indexing_family(meta, state))
}

#[inline(never)]
fn suspend_object<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_object_family(meta, state))
}

#[inline(never)]
fn suspend_arithmetic<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<arithmetic::ArithmeticDispatchFuture<'a>, RuntimeError> {
    arithmetic::suspend_arithmetic(meta.instr, state.stack, &state.context.runtime)
}

#[inline(never)]
fn suspend_output<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_output(meta, state))
}

#[inline(never)]
fn suspend_logical<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_logical(meta, state))
}

#[inline(never)]
fn suspend_aggregate<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_aggregate(meta, state))
}

async fn dispatch_parallel(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    let handled = parallel::dispatch(
        meta.instr,
        state.stack,
        state.vars,
        state.pc,
        state.sequence_register,
        parallel::ParallelDispatchContext {
            bytecode: meta.bytecode,
            execution: state.context,
            function_registry: meta.function_registry,
            current_function_name: meta.current_function_name,
        },
    )
    .await?;
    Ok(handled)
}

async fn dispatch_prepared_output(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> DispatchResult {
    let handled = prepared_outputs::dispatch(
        meta.instr,
        state.stack,
        state.vars,
        state.sequence_register,
        prepared_outputs::PreparedOutputContext {
            authority: slot_authority::AssignmentSlotAuthority {
                context: state.context,
                global_aliases: state.global_aliases,
                persistent_aliases: state.persistent_aliases,
                missing_input_slots: state.missing_input_slots,
                var_names: meta.var_names,
                current_function_name: meta.current_function_name,
            },
        },
        hooks.store_var_before_overwrite,
        hooks.store_var_after_store,
    )
    .await?;
    debug_assert!(handled, "prepared-output route must handle its instruction");
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

fn dispatch_call<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Pin<Box<dyn Future<Output = DispatchResult> + 'a>> {
    calls::dispatch_call(calls::CallDispatchContext {
        instr: meta.instr,
        stack: state.stack,
        vars: state.vars,
        sequence_register: state.sequence_register,
        context: state.context,
        try_stack: state.try_stack,
        last_exception: state.last_exception,
        imports: state.imports,
        pc: state.pc,
        source_id: meta.source_id,
        source_ref: meta
            .bytecode
            .source_id
            .and_then(crate::BytecodeSourceRef::runtime),
        call_arg_spans: meta.call_arg_spans,
        invocation_arity: meta.invocation_arity,
        function_registry: meta.function_registry,
        current_function_name: meta.current_function_name,
    })
}

async fn dispatch_indexing_family(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
) -> DispatchResult {
    let handled = indexing::dispatch_indexing(
        meta.instr,
        state.stack,
        state.vars,
        meta.function_registry,
        *state.pc,
        &state.context.runtime,
        state.sequence_register,
    )
    .await?;
    debug_assert!(handled, "indexing route must handle its instruction");
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn dispatch_object_family(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
) -> DispatchResult {
    let handled = object::dispatch_object(
        meta.instr,
        state.stack,
        state.sequence_register,
        object::ObjectDispatchContext {
            vars: state.vars,
            runtime: &state.context.runtime,
            current_function_name: meta.current_function_name,
            bytecode: meta.bytecode,
            invocation_arity: meta.invocation_arity,
        },
    )
    .await?;
    debug_assert!(handled, "object route must handle its instruction");
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn dispatch_output(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    match meta.instr {
        Instr::EmitStackTop { label } => {
            emit_stack_top(state.context, state.stack, label, meta.var_names).await?
        }
        Instr::EmitVar { var_index, label } => {
            emit_var(state.context, state.vars, *var_index, label, meta.var_names).await?
        }
        _ => unreachable!("output route received another instruction"),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn dispatch_logical(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    let decision = match meta.instr {
        Instr::AndAnd(target) => crate::ops::control_flow::and_and(
            logical_truth_from_value(
                &crate::interpreter::stack::pop_value(state.stack)?,
                "short-circuit && condition",
            )
            .await?,
            *target,
        ),
        Instr::OrOr(target) => crate::ops::control_flow::or_or(
            logical_truth_from_value(
                &crate::interpreter::stack::pop_value(state.stack)?,
                "short-circuit || condition",
            )
            .await?,
            *target,
        ),
        Instr::JumpIfFalse(target) => {
            let condition = crate::interpreter::stack::pop_value(state.stack)?;
            crate::ops::control_flow::jump_if_false(
                runmat_runtime::condition::control_flow_truth_from_value(
                    &state.context.runtime,
                    &condition,
                    "if condition",
                )
                .await?,
                *target,
            )
        }
        _ => unreachable!("logical route received another instruction"),
    };
    Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
        decision, state.pc,
    ))))
}

async fn dispatch_aggregate(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    let source_ref = meta
        .bytecode
        .source_id
        .and_then(crate::BytecodeSourceRef::runtime);
    let invocation_arity = meta.invocation_arity;
    match meta.instr {
        Instr::CreateMatrixFromSequences {
            row_lengths,
            elements,
            ..
        } => {
            let rows = materialize_aggregate_rows(
                state.stack,
                state.sequence_register,
                row_lengths,
                elements,
            )?;
            let _rows_roots = crate::call::root_values(rows.iter().flatten())?;
            runmat_runtime::create_matrix_from_values(&rows)
                .await
                .map_err(|error| {
                    arrays::qualify_matrix_construction_shape(error, source_ref, invocation_arity)
                })?
                .with_owned(|value| state.stack.try_push(value))?;
        }
        Instr::CreateMatrixDynamic(num_rows) => {
            create_matrix_dynamic(state.stack, *num_rows, |rows_data| async move {
                runmat_runtime::create_matrix_from_values(&rows_data)
                    .await
                    .map_err(|error| {
                        arrays::qualify_matrix_construction_shape(
                            error,
                            source_ref,
                            invocation_arity,
                        )
                    })
            })
            .await?;
        }
        Instr::CreateRange(has_step) => {
            let runtime = &state.context.runtime;
            create_range(state.stack, *has_step, |args| async move {
                crate::call::call_builtin_async(runtime, "colon", &args).await
            })
            .await?;
        }
        _ => unreachable!("aggregate route received another instruction"),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

#[cfg(test)]
#[path = "future/tests.rs"]
mod tests;
