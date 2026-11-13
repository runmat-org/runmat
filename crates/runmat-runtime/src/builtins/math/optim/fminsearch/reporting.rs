use super::*;
pub(super) use common::nelder_mead::{Procedure, Reason};
use runmat_builtins::OptimDisplay;
pub(super) fn initial(
    display: OptimDisplay,
    point: &evaluation::Point,
    calls: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if display != OptimDisplay::Iter {
        return Ok(());
    }
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        " Iteration  Func-count             f(x)  Procedure",
    )?;
    row(point, 0, calls, Procedure::Initial)
}
pub(super) fn iteration(
    display: OptimDisplay,
    point: &evaluation::Point,
    iteration: usize,
    calls: usize,
    procedure: Procedure,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if display != OptimDisplay::Iter {
        return Ok(());
    }
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    row(point, iteration, calls, procedure)
}
fn row(
    point: &evaluation::Point,
    iteration: usize,
    calls: usize,
    procedure: Procedure,
) -> BuiltinResult<()> {
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "{iteration:>10}  {calls:>10}  {:>15.7e}  {}",
            point.f,
            procedure.name()
        ),
    )
}
pub(super) fn finish(
    display: OptimDisplay,
    reason: Reason,
    iterations: usize,
    calls: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<String> {
    let description=match reason {Reason::Converged=>"The evaluated simplex satisfies both TolX and TolFun; Nelder-Mead does not certify a local or global minimum.",Reason::Iterations=>"Maximum number of simplex iterations reached before convergence.",Reason::Evaluations=>"Maximum number of objective evaluations reached before convergence.",Reason::Observer=>"Stopped by an output or plot function.",Reason::Time=>"Time limit reached before convergence."};
    let message = format!("{description} Iterations: {iterations}, FuncCount: {calls}.");
    if matches!(display, OptimDisplay::Final | OptimDisplay::Iter)
        || display == OptimDisplay::Notify && reason.flag() <= 0.0
    {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("fminsearch: {message}"),
        )?;
    }
    Ok(message)
}
