//! MATLAB-compatible `fzero` builtin for scalar nonlinear root finding.
//!
//! `fzero` searches for a scalar zero from either a two-point bracket or a
//! scalar initial guess.  It supports MATLAB's four output arities:
//!
//! * `x = fzero(fun, x0)`
//! * `x = fzero(fun, x0, options)`
//! * `[x, fval] = fzero(...)`
//! * `[x, fval, exitflag] = fzero(...)`
//! * `[x, fval, exitflag, output] = fzero(...)`

use runmat_builtins::{
    BuiltinErrorDescriptor, FZERO_CALLBACK_NUMERIC_EXTENSION, FZERO_ERROR_INVALID_ARGUMENT,
    FZERO_ERROR_INVALID_INPUT, FZERO_ERROR_TOO_MANY_OUTPUTS, FZERO_INPUT_NUMERIC_EXTENSION,
    FZERO_OPTION_EXTENSION, FZERO_RESIDENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::brent::{
    brent_zero, BrentParams, BrentZeroBracket, BrentZeroObserver, BrentZeroResult,
    BrentZeroStepKind,
};
use crate::builtins::math::optim::common::{
    call_scalar_function_with_precision, call_scalar_function_with_precision_info, option_f64,
    option_string, option_usize,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fzero";
const ALGORITHM: &str = "bisection, interpolation";
const DEFAULT_TOL_X: f64 = f64::EPSILON;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS: usize = 500;

fn fzero_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fzero:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fzero_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = fzero_error_with_detail(fallback, err.message());
        mapped.span = err.span.clone();
        mapped.context = err.context.clone();
        mapped.context.builtin = Some(NAME.to_string());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 4) {
        return Err(fzero_too_many_outputs_error());
    }
    Ok(())
}

fn fzero_too_many_outputs_error() -> RuntimeError {
    fzero_error_with_detail(
        &FZERO_ERROR_TOO_MANY_OUTPUTS,
        "fzero: too many output arguments; maximum is 4",
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fzero")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fzero",
    op_kind: GpuOpKind::Custom("scalar-root-find"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host iterative solver. Callback values may use GPU-aware builtins, but the root search runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fzero")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fzero",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Root finding repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fzero",
    binding_variant = "default",
    execution_stack = "process",
    builtin_path = "crate::builtins::math::optim::fzero"
)]
async fn fzero_builtin(function: Value, x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    validate_requested_outputs()?;
    let options = parse_options(rest.first())
        .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_ARGUMENT))?;
    let mut opts = FzeroOptions::from_struct(options.as_ref())
        .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_ARGUMENT))?;
    let x = crate::builtins::math::optim::common::prepare_floating_value(
        NAME,
        x,
        &FZERO_INPUT_NUMERIC_EXTENSION,
        &FZERO_RESIDENT_EXTENSION,
        "initial point or interval",
    )
    .await
    .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_INPUT))?;
    let initial_single = matches!(&x, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32);
    let bracket = initial_bracket(&function, x, &opts, initial_single)
        .await
        .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_INPUT))?;
    if !opts.tol_x_explicit {
        opts.tol_x = default_tol_x(initial_single || bracket.function_single);
    }
    let mut iter_log = IterDisplay::new(opts.display);
    let observer: Option<&mut dyn BrentZeroObserver> = if matches!(opts.display, DisplayMode::Iter)
    {
        Some(&mut iter_log)
    } else {
        None
    };
    let result = brent_zero(
        NAME,
        &function,
        BrentZeroBracket {
            a: bracket.a,
            b: bracket.b,
            fa: bracket.fa,
            fb: bracket.fb,
            evals: bracket.evals,
        },
        BrentParams {
            tol_x: opts.tol_x,
            max_iter: opts.max_iter,
            max_fun_evals: opts.max_fun_evals,
        },
        initial_single,
        observer,
    )
    .await
    .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_INPUT))?;
    finalize(result, &opts, initial_single)
}

fn default_tol_x(single: bool) -> f64 {
    if single {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    }
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy)]
struct FzeroOptions {
    tol_x: f64,
    tol_x_explicit: bool,
    max_iter: usize,
    max_fun_evals: usize,
    display: DisplayMode,
}

impl FzeroOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        crate::builtins::math::optim::common::ensure_option_extensions(
            NAME,
            options,
            &FZERO_OPTION_EXTENSION,
            &FZERO_RESIDENT_EXTENSION,
        )?;
        let display = DisplayMode::parse(&option_string(options, "Display", "off")?)?;
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        if tol_x <= 0.0 {
            return Err(fzero_error_with_detail(
                &FZERO_ERROR_INVALID_ARGUMENT,
                "option TolX must be positive",
            ));
        }
        let max_iter = option_usize(NAME, options, "MaxIter", DEFAULT_MAX_ITER)?;
        let max_fun_evals = option_usize(NAME, options, "MaxFunEvals", DEFAULT_MAX_FUN_EVALS)?;
        Ok(Self {
            tol_x,
            tol_x_explicit: options.is_some_and(|options| {
                options
                    .fields
                    .keys()
                    .any(|key| key.eq_ignore_ascii_case("TolX"))
            }),
            max_iter: max_iter.max(1),
            max_fun_evals: max_fun_evals.max(1),
            display,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayMode {
    Off,
    Iter,
    Final,
}

impl DisplayMode {
    fn parse(text: &str) -> BuiltinResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(Self::Off),
            "iter" => Ok(Self::Iter),
            "final" => Ok(Self::Final),
            other => Err(fzero_error_with_detail(
                &FZERO_ERROR_INVALID_ARGUMENT,
                format!("option Display must be 'off', 'none', 'final', or 'iter', got '{other}'"),
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct Bracket {
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    evals: usize,
    function_single: bool,
}

async fn initial_bracket(
    function: &Value,
    x: Value,
    options: &FzeroOptions,
    single: bool,
) -> BuiltinResult<Bracket> {
    let x = crate::dispatcher::gather_if_needed_async(&x).await?;
    match x {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(&tensor);
            match values.as_slice() {
                [a, b] => bracket_from_endpoints(function, *a, *b, single).await,
                [guess] => expand_bracket(function, *guess, options, single).await,
                _ => Err(fzero_error_with_detail(
                    &FZERO_ERROR_INVALID_INPUT,
                    "initial point must be a scalar or two-element bracket",
                )),
            }
        }
        Value::Num(n) => expand_bracket(function, n, options, single).await,
        Value::Int(i) => expand_bracket(function, i.to_f64(), options, single).await,
        Value::Bool(b) => {
            expand_bracket(function, if b { 1.0 } else { 0.0 }, options, single).await
        }
        other => Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            format!("initial point must be a scalar or two-element bracket, got {other:?}"),
        )),
    }
}

async fn bracket_from_endpoints(
    function: &Value,
    a: f64,
    b: f64,
    single: bool,
) -> BuiltinResult<Bracket> {
    if !a.is_finite() || !b.is_finite() || a == b {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "bracket endpoints must be finite and distinct",
        ));
    }
    let (fa, fa_single) =
        call_scalar_function_with_precision_info(NAME, function, a, single).await?;
    if fa == 0.0 {
        return Ok(Bracket {
            a,
            b: a,
            fa,
            fb: fa,
            evals: 1,
            function_single: fa_single,
        });
    }
    let (fb, fb_single) =
        call_scalar_function_with_precision_info(NAME, function, b, single).await?;
    if fb == 0.0 || fa.signum() != fb.signum() {
        Ok(Bracket {
            a,
            b,
            fa,
            fb,
            evals: 2,
            function_single: fa_single || fb_single,
        })
    } else {
        Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "function values at bracket endpoints must differ in sign",
        ))
    }
}

async fn expand_bracket(
    function: &Value,
    x0: f64,
    options: &FzeroOptions,
    single: bool,
) -> BuiltinResult<Bracket> {
    if !x0.is_finite() {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "initial point must be finite",
        ));
    }
    let (f0, function_single) =
        call_scalar_function_with_precision_info(NAME, function, x0, single).await?;
    if f0 == 0.0 {
        return Ok(Bracket {
            a: x0,
            b: x0,
            fa: f0,
            fb: f0,
            evals: 1,
            function_single,
        });
    }

    let mut evals = 1usize;
    let mut step = (x0.abs() * 0.01).max(0.01);
    while evals + 2 <= options.max_fun_evals {
        let a = x0 - step;
        let b = x0 + step;
        let fa = call_scalar_function_with_precision(NAME, function, a, single).await?;
        let fb = call_scalar_function_with_precision(NAME, function, b, single).await?;
        evals += 2;
        if fa == 0.0 {
            return Ok(Bracket {
                a,
                b: a,
                fa,
                fb: fa,
                evals,
                function_single,
            });
        }
        if fa.signum() != f0.signum() {
            return Ok(Bracket {
                a,
                b: x0,
                fa,
                fb: f0,
                evals,
                function_single,
            });
        }
        if fb.signum() != f0.signum() {
            return Ok(Bracket {
                a: x0,
                b,
                fa: f0,
                fb,
                evals,
                function_single,
            });
        }
        if fb == 0.0 || fa.signum() != fb.signum() {
            return Ok(Bracket {
                a,
                b,
                fa,
                fb,
                evals,
                function_single,
            });
        }
        step *= 1.6;
    }

    Err(fzero_error_with_detail(
        &FZERO_ERROR_INVALID_INPUT,
        "could not find a sign-changing bracket around the initial point",
    ))
}

fn finalize(result: BrentZeroResult, options: &FzeroOptions, single: bool) -> BuiltinResult<Value> {
    let exit_flag = if result.converged { 1 } else { 0 };
    let message = build_message(&result);
    emit_summary(&result, exit_flag, &message, options);

    let scalar = |value: f64| -> BuiltinResult<Value> {
        if single {
            Tensor::new_with_dtype(vec![value], vec![1, 1], runmat_value::NumericDType::F32)
                .map(Value::Tensor)
                .map_err(|error| fzero_error_with_detail(&FZERO_ERROR_INVALID_INPUT, error))
        } else {
            Ok(Value::Num(value))
        }
    };
    let x = scalar(result.x)?;
    let fval = scalar(result.fval)?;
    let exitflag = scalar(exit_flag as f64)?;
    let output_struct = Value::Struct(build_output_struct(&result, &message));

    match crate::output_count::current_output_count() {
        None => Ok(x),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(crate::output_count::output_list_with_padding(1, vec![x])),
        Some(2) => Ok(crate::output_count::output_list_with_padding(
            2,
            vec![x, fval],
        )),
        Some(3) => Ok(crate::output_count::output_list_with_padding(
            3,
            vec![x, fval, exitflag],
        )),
        Some(4) => Ok(crate::output_count::output_list_with_padding(
            4,
            vec![x, fval, exitflag, output_struct],
        )),
        Some(_) => Err(fzero_too_many_outputs_error()),
    }
}

fn build_output_struct(result: &BrentZeroResult, message: &str) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("message", Value::from(message.to_string()));
    fields
}

fn build_message(result: &BrentZeroResult) -> String {
    if result.converged {
        format!(
            "Zero found within OPTIONS.TolX. Iterations: {}, FuncCount: {}.",
            result.iterations, result.func_count
        )
    } else {
        format!(
            "Exiting: Maximum number of function evaluations or iterations has been exceeded - increase MaxFunEvals or MaxIter. Iterations: {}, FuncCount: {}.",
            result.iterations, result.func_count
        )
    }
}

fn emit_summary(result: &BrentZeroResult, exit_flag: i32, message: &str, options: &FzeroOptions) {
    if !matches!(options.display, DisplayMode::Final | DisplayMode::Iter) {
        return;
    }
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "fzero: x = {x:.6}, fval = {fval:.6}, exitflag = {exit_flag}. {message}",
            x = result.x,
            fval = result.fval,
        ),
    );
}

struct IterDisplay {
    mode: DisplayMode,
    printed_header: bool,
}

impl IterDisplay {
    fn new(mode: DisplayMode) -> Self {
        Self {
            mode,
            printed_header: false,
        }
    }
}

impl BrentZeroObserver for IterDisplay {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentZeroStepKind,
    ) {
        if !matches!(self.mode, DisplayMode::Iter) {
            return;
        }
        if !self.printed_header {
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                " Func-count        x          f(x)          Procedure",
            );
            self.printed_header = true;
        }
        let procedure = match step_kind {
            BrentZeroStepKind::Initial => "initial",
            BrentZeroStepKind::Bisection => "bisection",
            BrentZeroStepKind::Interpolation => "interpolation",
        };
        let line =
            format!("    {func_count:>5}    {x:13.6e} {fx:13.6e}    {procedure}    (iter {iter})");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line);
    }
}

#[cfg(test)]
#[path = "fzero/tests.rs"]
mod tests;
