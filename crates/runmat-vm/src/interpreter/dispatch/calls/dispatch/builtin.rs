//! Select one concrete builtin, superclass, or feval future before polling.
//!
//! The call-family router already owns one box for a suspending instruction.
//! Keeping these unrelated await states out of one async match prevents the
//! leaf builtin frame from inheriting every superclass and feval branch.

use super::*;

#[path = "builtin/feval.rs"]
mod feval;
#[path = "builtin/ordinary.rs"]
mod ordinary;
#[path = "builtin/superclass.rs"]
mod superclass;

pub(super) fn dispatch<'a>(ctx: CallDispatchContext<'a>) -> CallFuture<'a> {
    match ctx.instr {
        Instr::CallBuiltinMulti(..) | Instr::CallBuiltinMultiUsingOutputSlot(..) => {
            suspend_ordinary(ctx)
        }
        Instr::CallSuperConstructorMulti { .. } | Instr::CallSuperMethodMulti { .. } => {
            suspend_superclass(ctx)
        }
        Instr::CallFevalMulti(..)
        | Instr::CallFevalMultiUsingOutputSlot(..)
        | Instr::CallFevalExpandMultiOutput(..)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..) => suspend_feval(ctx),
        _ => unreachable!("builtin-family router received another instruction"),
    }
}

#[inline(never)]
fn suspend_ordinary(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || ordinary::dispatch(ctx))
}

#[inline(never)]
fn suspend_superclass(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || superclass::dispatch(ctx))
}

#[inline(never)]
fn suspend_feval(ctx: CallDispatchContext<'_>) -> CallFuture<'_> {
    let context = ctx.context;
    control::admit(&context.runtime, move || feval::dispatch(ctx))
}
