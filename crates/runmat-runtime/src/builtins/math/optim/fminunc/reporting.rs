//! Presentation and terminal identities for the existing BFGS state machine.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum TerminationReason {
    Optimality,
    Step,
    Limit,
    OutputFunction,
    NumericalFailure,
    ObjectiveLimit,
}
impl TerminationReason {
    pub(super) fn flag(self) -> i32 {
        match self {
            Self::Optimality => 1,
            Self::Step => 2,
            Self::Limit => 0,
            Self::OutputFunction => -1,
            Self::NumericalFailure => -4,
            Self::ObjectiveLimit => -3,
        }
    }
    fn description(self) -> &'static str {
        match self {
            Self::Optimality => "First-order optimality is below OptimalityTolerance (legacy TolFun).",
            Self::Step => "Step displacement is below StepTolerance (legacy TolX). First-order optimality need not be satisfied.",
            Self::Limit => "Iteration or function-evaluation limit reached before a stopping tolerance was satisfied.",
            Self::OutputFunction => "Optimization stopped by the output function.",
            Self::ObjectiveLimit => "The evaluated objective reached ObjectiveLimit. This is a threshold stop, not a proof of unboundedness.",
            Self::NumericalFailure => "RunMat numerical failure: line search could not produce an acceptable step; no local-minimum claim is made.",
        }
    }
}

pub(super) struct Progress {
    pub iteration: usize,
    pub func_count: usize,
    pub fval: f64,
    pub firstorderopt: f64,
    pub line_step: Option<f64>,
}

pub(super) fn initial(
    display: DisplayMode,
    progress: Progress,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if display != DisplayMode::Iter {
        return Ok(());
    }
    // Console capture, diary and forwarding must use the original invocation,
    // not a replacement ambient context installed by an objective/observer.
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        " Iteration  Func-count             f(x)     Step-size  First-order optimality",
    )?;
    row(progress)
}

pub(super) fn iteration(
    display: DisplayMode,
    progress: Progress,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if display != DisplayMode::Iter {
        return Ok(());
    }
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    row(progress)
}

fn row(progress: Progress) -> BuiltinResult<()> {
    let step = progress
        .line_step
        .map(|step| format!("{step:.6e}"))
        .unwrap_or_else(|| "-".into());
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "{:>10}  {:>10}  {:>15.6e}  {:>12}  {:>22.6e}",
            progress.iteration, progress.func_count, progress.fval, step, progress.firstorderopt,
        ),
    )
}

pub(super) fn finish(
    display: DisplayMode,
    reason: TerminationReason,
    progress: Progress,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<String> {
    if reason == TerminationReason::NumericalFailure {
        let extension = &runmat_builtins::FMINUNC_NUMERICAL_FAILURE_EXTENSION;
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime, extension, NAME,
        )?;
    }
    let message = format!(
        "{} Iterations: {}, FuncCount: {}.",
        reason.description(),
        progress.iteration,
        progress.func_count,
    );
    let emit = match display {
        DisplayMode::Off | DisplayMode::None => false,
        DisplayMode::Final | DisplayMode::Iter => true,
        DisplayMode::Notify => reason.flag() <= 0,
        DisplayMode::Diagnose => {
            let mut error = fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "diagnostic display is not supported by fminunc",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
    };
    if emit {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!(
                "fminunc: fval = {:.6e}, firstorderopt = {:.6e}, exitflag = {}. {message}",
                progress.fval,
                progress.firstorderopt,
                reason.flag(),
            ),
        )?;
    }
    Ok(message)
}

pub(super) fn trust_region_fallback(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    crate::context::with_original_scope(runtime, || {
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: NAME,
            identifier: "RunMat:fminunc:TrustRegionRequiresGradient",
            message: "The trust-region algorithm requires a supplied objective gradient; using quasi-newton instead. Trust-region-only settings are inactive.",
            show_identifier: true,
        })
    })
}
