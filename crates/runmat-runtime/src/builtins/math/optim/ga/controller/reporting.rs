//! Synchronous console writes enter only the selected original context.
use super::{
    super::{inputs::Problem, options::Options, outcome::Termination, ScalarCooperation},
    search::Search,
};
use crate::BuiltinResult;
use runmat_builtins::OptimDisplay;
fn write(line: String, cooperation: &ScalarCooperation) -> BuiltinResult<()> {
    let _scope = cooperation
        .runtime()
        .map(crate::context::RuntimeContext::enter);
    crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line)
}
pub(super) fn begin(
    problem: &Problem,
    options: &Options,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    if options.display == OptimDisplay::Diagnose {
        write(format!("ga: {} variables; {} population members in {} subpopulations; population {}; nonlinear algorithm {}; vectorized {}; parallel {:?}; integer constraints {}",
            problem.variables, options.layout.total(), options.layout.groups(), options.population.name(),
            options.constraint.name(), options.vectorized, options.parallel, problem.mixed_integer), cooperation)?;
    }
    if matches!(options.display, OptimDisplay::Iter | OptimDisplay::Diagnose) {
        write(
            "Generation  Objective points  Nonlinear points  Best feasible fitness".into(),
            cooperation,
        )?;
    }
    Ok(())
}
pub(super) fn iteration<G>(
    search: &Search<G>,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    if matches!(
        search.options.display,
        OptimDisplay::Iter | OptimDisplay::Diagnose
    ) && search
        .progress
        .generation
        .is_multiple_of(search.options.display_interval)
    {
        write(
            format!(
                "{}  {}  {}  {}",
                search.progress.generation,
                search.objective_counts.points,
                search.nonlinear_counts.points,
                search
                    .best
                    .score()
                    .map_or_else(|| "undefined".into(), |value| format!("{value:.8e}"))
            ),
            cooperation,
        )?;
    }
    Ok(())
}
pub(super) fn finish<G>(
    search: &Search<G>,
    reason: Termination,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    stopped(search.options.display, reason, cooperation)
}
pub(super) fn stopped(
    display: OptimDisplay,
    reason: Termination,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    if !matches!(display, OptimDisplay::Off | OptimDisplay::None) {
        write(format!("ga: {}", reason.message()), cooperation)?;
    }
    Ok(())
}
