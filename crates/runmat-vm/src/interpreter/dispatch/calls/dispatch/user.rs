//! Select one semantic user-call future before polling a Process frame.
//!
//! The call-family router already owns one box for a suspending instruction.
//! Keep ordinary and captured-nested async variants in separate concrete
//! futures so their unrelated await states do not accumulate on each native
//! recursive poll frame.

use super::*;

#[path = "user/ordinary.rs"]
mod ordinary;
#[path = "user/nested.rs"]
mod nested;

pub(super) fn dispatch<'a>(ctx: CallDispatchContext<'a>) -> CallFuture<'a> {
    match ctx.instr {
        Instr::CallSemanticFunctionMulti(..)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(..)
        | Instr::CallFunctionMulti { .. }
        | Instr::CallFunctionMultiUsingOutputSlot { .. } => suspend_ordinary(ctx),
        Instr::CallSemanticNestedFunctionMulti { .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. } => suspend_nested(ctx),
        _ => unreachable!("user-call router received another instruction"),
    }
}

#[inline(never)]
fn suspend_ordinary(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    Box::pin(ordinary::dispatch(ctx))
}

#[inline(never)]
fn suspend_nested(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    Box::pin(nested::dispatch(ctx))
}
