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
    immediate: Option<ImmediateInstruction<'a>>,
    state: InstructionDispatchFutureState<'a>,
}

enum InstructionDispatchFutureState<'a> {
    Immediate,
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
            InstructionDispatchFutureState::Immediate => {
                let instruction = this
                    .immediate
                    .take()
                    .expect("completed instruction future was polled again");
                Poll::Ready(dispatch_immediate(
                    instruction.meta,
                    instruction.state,
                    instruction.hooks,
                ))
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
            immediate: Some(ImmediateInstruction { meta, state, hooks }),
            state: InstructionDispatchFutureState::Immediate,
        }
    }

    fn suspended(future: SuspendedDispatchFuture<'a>) -> Self {
        Self {
            immediate: None,
            state: InstructionDispatchFutureState::Suspended(future),
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
        InstructionDispatchFamily::Arithmetic => {
            InstructionDispatchFuture::suspended(suspend_arithmetic(meta, state))
        }
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
) -> SuspendedDispatchFuture<'a> {
    Box::pin(dispatch_arithmetic_family(meta, state))
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
        meta.current_function_name,
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

async fn dispatch_arithmetic_family(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
) -> DispatchResult {
    let handled =
        arithmetic::dispatch_arithmetic(meta.instr, state.stack, &state.context.runtime).await?;
    debug_assert!(handled, "arithmetic route must handle its instruction");
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}

async fn dispatch_output(meta: DispatchMeta<'_>, state: DispatchState<'_>) -> DispatchResult {
    match meta.instr {
        Instr::EmitStackTop { label } => emit_stack_top(state.stack, label, meta.var_names).await?,
        Instr::EmitVar { var_index, label } => {
            emit_var(state.vars, *var_index, label, meta.var_names).await?
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
                logical_truth_from_value(&condition, "if condition").await?,
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
            state
                .stack
                .push(runmat_runtime::create_matrix_from_values(&rows).await?);
        }
        Instr::CreateMatrixDynamic(num_rows) => {
            create_matrix_dynamic(state.stack, *num_rows, |rows_data| async move {
                runmat_runtime::create_matrix_from_values(&rows_data).await
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
