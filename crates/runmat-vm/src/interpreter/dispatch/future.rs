//! Poll-shape owner for one VM instruction.
//!
//! Immediate instructions complete without allocation. A selected instruction
//! family owns the only boxed future when the operation can actually suspend.

use super::*;

#[path = "future/name.rs"]
mod name;
#[path = "future/semantic.rs"]
mod semantic;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(super) type DispatchResult = Result<Option<DispatchHandled>, RuntimeError>;
#[path = "future/control.rs"]
pub(super) mod control;
pub(super) type AdmittedDispatchFuture<'a> =
    runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>;

pub(super) struct InstructionDispatchFuture<'a> {
    state: InstructionDispatchFutureState<'a>,
}

enum InstructionDispatchFutureState<'a> {
    Immediate(Option<ImmediateInstruction<'a>>),
    Ready(Option<DispatchResult>),
    StackMutation(AdmittedDispatchFuture<'a>),
    Control(
        runmat_runtime::execution::RuntimeContinuationFuture<
            'a,
            Option<runmat_runtime::execution::RootedResult<DispatchHandled>>,
        >,
    ),
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
            InstructionDispatchFutureState::StackMutation(future) => {
                match future.as_mut().poll(context) {
                    Poll::Ready(Ok(())) => Poll::Ready(Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )))),
                    Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
                    Poll::Pending => Poll::Pending,
                }
            }
            InstructionDispatchFutureState::Control(future) => {
                future.as_mut().poll(context).map(|result| {
                    result.map(|control| control.map(|control| control.with_owned(|value| value)))
                })
            }
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

    fn control(
        future: runmat_runtime::execution::RuntimeContinuationFuture<
            'a,
            Option<runmat_runtime::execution::RootedResult<DispatchHandled>>,
        >,
    ) -> Self {
        Self {
            state: InstructionDispatchFutureState::Control(future),
        }
    }

    fn ready(result: DispatchResult) -> Self {
        Self {
            state: InstructionDispatchFutureState::Ready(Some(result)),
        }
    }

    fn stack_mutation(future: AdmittedDispatchFuture<'a>) -> Self {
        Self {
            state: InstructionDispatchFutureState::StackMutation(future),
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
                admitted_instruction(suspend_semantic_closure(meta, state))
            } else {
                InstructionDispatchFuture::immediate(meta, state, hooks)
            }
        }
        InstructionDispatchFamily::NameHandle => {
            admitted_instruction(suspend_name_handle(meta, state))
        }
        InstructionDispatchFamily::Parallel => {
            controlled_instruction(suspend_parallel(meta, state))
        }
        InstructionDispatchFamily::PreparedOutput => {
            match suspend_prepared_output(meta, state, hooks) {
                Ok(future) => InstructionDispatchFuture::stack_mutation(future),
                Err(error) => InstructionDispatchFuture::ready(Err(error)),
            }
        }
        InstructionDispatchFamily::Call => controlled_instruction(dispatch_call(meta, state)),
        InstructionDispatchFamily::Indexing => admitted_instruction(suspend_indexing(meta, state)),
        InstructionDispatchFamily::Object => admitted_instruction(suspend_object(meta, state)),
        InstructionDispatchFamily::Arithmetic => match suspend_arithmetic(meta, state) {
            Ok(future) => InstructionDispatchFuture::stack_mutation(future),
            Err(error) => InstructionDispatchFuture::ready(Err(error)),
        },
        InstructionDispatchFamily::Output => admitted_instruction(suspend_output(meta, state)),
        InstructionDispatchFamily::Logical => controlled_instruction(suspend_logical(meta, state)),
        InstructionDispatchFamily::Aggregate => {
            admitted_instruction(suspend_aggregate(meta, state))
        }
    }
}

fn admitted_instruction<'a>(
    future: Result<AdmittedDispatchFuture<'a>, RuntimeError>,
) -> InstructionDispatchFuture<'a> {
    match future {
        Ok(future) => InstructionDispatchFuture::stack_mutation(future),
        Err(error) => InstructionDispatchFuture::ready(Err(error)),
    }
}

fn controlled_instruction<'a>(
    future: control::ControlDispatchFuture<'a>,
) -> InstructionDispatchFuture<'a> {
    match future {
        Ok(future) => InstructionDispatchFuture::control(future),
        Err(error) => InstructionDispatchFuture::ready(Err(error)),
    }
}

#[inline(never)]
fn suspend_semantic_closure<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let context = &*state.context;
    runmat_runtime::execution::admit_domain_continuation_with(&context.runtime, move || {
        semantic::dispatch(meta, state.stack, context)
    })
}

#[inline(never)]
fn suspend_name_handle<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let context = &*state.context;
    runmat_runtime::execution::admit_domain_continuation_with(&context.runtime, move || {
        name::dispatch(meta, state.stack, context)
    })
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
) -> control::ControlDispatchFuture<'a> {
    parallel::dispatch(
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
}

#[inline(never)]
fn suspend_prepared_output<'a>(
    mut meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let call_arg_spans = meta.call_arg_spans.take();
    prepared_outputs::dispatch(
        prepared_outputs::Request {
            instruction: meta.instr,
            stack: state.stack,
            vars: state.vars,
            state: state.sequence_register,
            context: prepared_outputs::PreparedOutputContext {
                bytecode: meta.bytecode,
                authority: slot_authority::AssignmentSlotAuthority {
                    context: state.context,
                    global_aliases: state.global_aliases,
                    persistent_aliases: state.persistent_aliases,
                    missing_input_slots: state.missing_input_slots,
                    var_names: meta.var_names,
                    current_function_name: meta.current_function_name,
                },
            },
            before_overwrite: hooks.store_var_before_overwrite,
            after_store: hooks.store_var_after_store,
        },
        move |result| {
            drop(call_arg_spans);
            let handled = result?;
            debug_assert!(handled, "prepared-output route must handle its instruction");
            Ok(())
        },
    )
}

#[inline(never)]
fn suspend_indexing<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let context = &*state.context;
    runmat_runtime::execution::admit_domain_continuation_with(&context.runtime, move || {
        dispatch_indexing_family(
            meta,
            state.stack,
            state.vars,
            state.sequence_register,
            *state.pc,
            &context.runtime,
        )
    })
}

#[inline(never)]
fn suspend_object<'a>(
    mut meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let call_arg_spans = meta.call_arg_spans.take();
    object::dispatch_object(
        object::Request {
            instruction: meta.instr,
            stack: state.stack,
            sequence_register: state.sequence_register,
            context: object::ObjectDispatchContext {
                vars: state.vars,
                runtime: &state.context.runtime,
                current_function_name: meta.current_function_name,
                bytecode: meta.bytecode,
                invocation_arity: meta.invocation_arity,
            },
        },
        move |result| {
            drop(call_arg_spans);
            let handled = result?;
            debug_assert!(handled, "object route must handle its instruction");
            Ok(())
        },
    )
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
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let context = &*state.context;
    runmat_runtime::execution::admit_domain_continuation_with(&context.runtime, move || {
        dispatch_output(meta, state.stack, state.vars, context)
    })
}

#[inline(never)]
fn suspend_logical<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> control::ControlDispatchFuture<'a> {
    let context = &*state.context;
    control::admit(&context.runtime, move || {
        dispatch_logical(meta, state.stack, state.pc, &context.runtime)
    })
}

#[inline(never)]
fn suspend_aggregate<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> Result<AdmittedDispatchFuture<'a>, RuntimeError> {
    let context = &*state.context;
    runmat_runtime::execution::admit_domain_continuation_with(&context.runtime, move || {
        dispatch_aggregate(meta, state.stack, state.sequence_register, context)
    })
}

fn dispatch_call<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
) -> control::ControlDispatchFuture<'a> {
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
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &mut InterpreterVars,
    sequence_register: &mut SequenceState,
    pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<(), RuntimeError> {
    let _call_arg_spans = meta.call_arg_spans.take();
    let handled = indexing::dispatch_indexing(
        meta.instr,
        stack,
        vars,
        meta.function_registry,
        pc,
        runtime,
        sequence_register,
    )
    .await?;
    debug_assert!(handled, "indexing route must handle its instruction");
    Ok(())
}

async fn dispatch_output(
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &mut InterpreterVars,
    context: &crate::bytecode::program::ExecutionContext,
) -> Result<(), RuntimeError> {
    let _call_arg_spans = meta.call_arg_spans.take();
    match meta.instr {
        Instr::EmitStackTop { label } => {
            emit_stack_top(context, stack, label, meta.var_names).await?
        }
        Instr::EmitVar { var_index, label } => {
            emit_var(context, vars, *var_index, label, meta.var_names).await?
        }
        _ => unreachable!("output route received another instruction"),
    }
    Ok(())
}

async fn dispatch_logical(
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    pc: &mut usize,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> DispatchResult {
    let _call_arg_spans = meta.call_arg_spans.take();
    let decision = match meta.instr {
        Instr::AndAnd(target) => crate::ops::control_flow::and_and(
            logical_truth_from_value(
                &crate::interpreter::stack::pop_value(stack)?,
                "short-circuit && condition",
            )
            .await?,
            *target,
        ),
        Instr::OrOr(target) => crate::ops::control_flow::or_or(
            logical_truth_from_value(
                &crate::interpreter::stack::pop_value(stack)?,
                "short-circuit || condition",
            )
            .await?,
            *target,
        ),
        Instr::JumpIfFalse(target) => {
            let condition = crate::interpreter::stack::pop_value(stack)?;
            crate::ops::control_flow::jump_if_false(
                runmat_runtime::condition::control_flow_truth_from_value(
                    runtime,
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
        decision, pc,
    ))))
}

async fn dispatch_aggregate(
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_register: &mut SequenceState,
    context: &crate::bytecode::program::ExecutionContext,
) -> Result<(), RuntimeError> {
    let _call_arg_spans = meta.call_arg_spans.take();
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
            let rows = materialize_aggregate_rows(stack, sequence_register, row_lengths, elements)?;
            let _rows_roots = crate::call::root_values(rows.iter().flatten())?;
            runmat_runtime::create_matrix_from_values(&rows)
                .await
                .map_err(|error| {
                    arrays::qualify_matrix_construction_shape(error, source_ref, invocation_arity)
                })?
                .with_owned(|value| stack.try_push(value))?;
        }
        Instr::CreateRange(has_step) => {
            let runtime = &context.runtime;
            create_range(stack, *has_step, |args| async move {
                crate::call::call_builtin_async(runtime, "colon", &args).await
            })
            .await?;
        }
        _ => unreachable!("aggregate route received another instruction"),
    }
    Ok(())
}

#[cfg(test)]
#[path = "future/tests.rs"]
mod tests;
