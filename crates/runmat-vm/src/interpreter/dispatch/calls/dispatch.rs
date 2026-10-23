//! One call-family boundary for VM instructions that can suspend. Each child
//! owns a bounded subset of the call grammar so its polling frame does not
//! become part of the general instruction dispatch frame.

use super::super::future::control;
use super::super::*;

type CallFuture<'a> = control::ControlDispatchFuture<'a>;

#[derive(Clone, Copy)]
enum OutputPublication {
    Fixed(usize),
    Sequence,
}

#[path = "dispatch/builtin.rs"]
mod builtin;
#[path = "dispatch/expanded.rs"]
mod expanded;
#[path = "dispatch/method.rs"]
mod method;
#[path = "dispatch/user.rs"]
mod user;
#[path = "dispatch/workspace.rs"]
mod workspace;

pub(in crate::interpreter::dispatch) struct CallDispatchContext<'a> {
    pub instr: &'a Instr,
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub vars: &'a mut crate::interpreter::vars::InterpreterVars,
    pub sequence_register: &'a mut SequenceState,
    pub context: &'a crate::bytecode::program::ExecutionContext,
    pub try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub last_exception: &'a mut Option<runmat_value::MException>,
    pub imports: &'a Vec<(Vec<String>, bool)>,
    pub pc: &'a mut usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    pub call_arg_spans: Option<runmat_runtime::callsite::CallsiteSpans>,
    pub invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub current_function_name: &'a str,
}

pub(in crate::interpreter::dispatch) fn dispatch_call<'a>(
    ctx: CallDispatchContext<'a>,
) -> CallFuture<'a> {
    match ctx.instr {
        Instr::CallBuiltinMulti(..)
        | Instr::CallBuiltinMultiUsingOutputSlot(..)
        | Instr::CallSuperConstructorMulti { .. }
        | Instr::CallSuperMethodMulti { .. }
        | Instr::CallFevalMulti(..)
        | Instr::CallFevalMultiUsingOutputSlot(..)
        | Instr::CallFevalExpandMultiOutput(..)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..) => suspend_builtin(ctx),
        Instr::CallSemanticFunctionMulti(..)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(..)
        | Instr::CallSemanticNestedFunctionMulti { .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
        | Instr::CallFunctionMulti { .. }
        | Instr::CallFunctionMultiUsingOutputSlot { .. } => suspend_user(ctx),
        Instr::CallWorkspaceFirstMulti { .. }
        | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. } => suspend_workspace(ctx),
        Instr::CallBuiltinExpandMultiOutput(..)
        | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(..)
        | Instr::CallSuperConstructorExpandMultiOutput { .. }
        | Instr::CallSuperMethodExpandMultiOutput { .. }
        | Instr::CallFunctionExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
        | Instr::CallSemanticFunctionExpandMultiOutput(..)
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. } => suspend_expanded(ctx),
        Instr::CallMethodOrMemberIndexMulti { .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. } => suspend_method(ctx),
        _ => unreachable!("call dispatcher received a non-call instruction"),
    }
}

// As with the top-level instruction router, isolate concrete future
// construction from this exhaustive match. Otherwise an unoptimized build
// reserves the largest call-family future whenever it selects even a small
// semantic-call child, once for every live Process frame.
#[inline(never)]
fn suspend_builtin(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    builtin::dispatch(ctx)
}

#[inline(never)]
fn suspend_user(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    user::dispatch(ctx)
}

#[inline(never)]
fn suspend_workspace(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || workspace::dispatch(ctx))
}

#[inline(never)]
fn suspend_expanded(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || expanded::dispatch(ctx))
}

#[inline(never)]
fn suspend_method(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || method::dispatch(ctx))
}
