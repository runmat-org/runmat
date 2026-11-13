//! Original-owner optional residual progress; off mode does not format a line.
use super::*;
use crate::context::RuntimeContext;
use runmat_builtins::OptimDisplay;
pub(super) enum Event {
    Iteration,
    Final,
}
pub(super) fn emit(
    runtime: Option<&RuntimeContext>,
    display: OptimDisplay,
    event: Event,
    line: impl FnOnce() -> String,
) -> BuiltinResult<()> {
    let enabled = match event {
        Event::Iteration => display == OptimDisplay::Iter,
        Event::Final => matches!(display, OptimDisplay::Iter | OptimDisplay::Final),
    };
    if enabled {
        crate::context::with_original_scope(runtime, || {
            crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line())
        })?;
    }
    Ok(())
}
pub(super) fn finish(
    runtime: Option<RuntimeContext>,
    display: OptimDisplay,
    result: LeastSquaresResult,
) -> BuiltinResult<LeastSquaresResult> {
    emit(runtime.as_ref(), display, Event::Final, || {
        format!(
            "lsqnonlin: {} iterations={} funcCount={}",
            result.message, result.iterations, result.func_count,
        )
    })?;
    Ok(result)
}
