//! MATLAB-compatible `fminbnd` builtin for bounded scalar minimization.
//!
//! `fminbnd` finds a local minimum of a scalar function on a finite interval
//! using Brent's method (golden-section search combined with parabolic
//! interpolation).  The implementation supports MATLAB's four output arities:
//!
//! * `x = fminbnd(fun, x1, x2)`
//! * `x = fminbnd(fun, x1, x2, options)`
//! * `[x, fval] = fminbnd(...)`
//! * `[x, fval, exitflag] = fminbnd(...)`
//! * `[x, fval, exitflag, output] = fminbnd(...)`
//!
//! The optional options struct (typically created by `optimset`) honours
//! `TolX`, `MaxIter`, `MaxFunEvals`, and `Display`.

use runmat_builtins::{
    BuiltinErrorDescriptor, OptimDisplay, OptimOptionField, OptimSolver,
    FMINBND_ERROR_INVALID_ARGUMENT, FMINBND_ERROR_INVALID_INPUT, FMINBND_ERROR_TOO_MANY_OUTPUTS,
    FMINBND_INPUT_NUMERIC_EXTENSION, FMINBND_OPTION_EXTENSION, FMINBND_RESIDENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::brent::{
    brent_min, BrentMinObserver, BrentMinResult, BrentParams, BrentStepKind,
};
use crate::builtins::math::optim::common::{
    preflight_scalar_operand, ScalarCooperation, ScalarOperandGeometry, ScalarOptionFields,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fminbnd";
const ALGORITHM: &str = "golden section search, parabolic interpolation";
const DEFAULT_TOL_X: f64 = 1.0e-4;
const DEFAULT_MAX_ITER: usize = 500;
const DEFAULT_MAX_FUN_EVALS: usize = 500;
const DEFAULT_DISPLAY: OptimDisplay = OptimDisplay::Notify;

fn fminbnd_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fminbnd:") {
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

fn fminbnd_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = fminbnd_error_with_detail(fallback, err.message());
        mapped.span = err.span;
        mapped.context = err.context.clone();
        mapped.context.builtin = Some(NAME.to_string());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fminbnd")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fminbnd",
    op_kind: GpuOpKind::Custom("bounded-scalar-min"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host iterative solver. Callback computations may use GPU-aware builtins, but the minimization loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fminbnd")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fminbnd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Bounded scalar minimization repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fminbnd",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fminbnd"
)]
pub(super) async fn fminbnd_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    x1: Value,
    x2: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    fminbnd_in_runtime(runtime.as_ref(), function, x1, x2, rest).await
}

#[cfg(test)]
pub(super) async fn fminbnd_builtin(
    function: Value,
    x1: Value,
    x2: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    fminbnd_in_runtime(None, function, x1, x2, rest).await
}

async fn fminbnd_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    x1: Value,
    x2: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fminbnd, runtime);
    cooperation.completed(0).await?;
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 4) {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_TOO_MANY_OUTPUTS,
            "maximum is four outputs",
        ));
    }
    if rest.len() > 1 {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options_struct = parse_options(rest.into_iter().next())
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_ARGUMENT))?;
    let options = FminbndOptions::from_struct(options_struct.as_ref(), runtime)
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_ARGUMENT))?;
    crate::call::admission::admit_feval_target(&function)?;
    for (value, role) in [(&x1, "lower bound"), (&x2, "upper bound")] {
        preflight_scalar_operand(
            OptimSolver::Fminbnd,
            value,
            ScalarOperandGeometry::Scalar,
            role,
            &FMINBND_INPUT_NUMERIC_EXTENSION,
            &FMINBND_RESIDENT_EXTENSION,
            runtime,
        )
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    }
    let x1 = crate::builtins::math::optim::common::prepare_floating_value_with_cooperation(
        NAME,
        x1,
        &FMINBND_INPUT_NUMERIC_EXTENSION,
        &FMINBND_RESIDENT_EXTENSION,
        "lower bound",
        &mut cooperation,
    )
    .await
    .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    let x1 = scalar_bound("lower bound", x1)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    cooperation.completed(0).await?;
    let x2 = crate::builtins::math::optim::common::prepare_floating_value_with_cooperation(
        NAME,
        x2,
        &FMINBND_INPUT_NUMERIC_EXTENSION,
        &FMINBND_RESIDENT_EXTENSION,
        "upper bound",
        &mut cooperation,
    )
    .await
    .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    let x2 = scalar_bound("upper bound", x2)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    cooperation.completed(0).await?;

    if !x1.is_finite() || !x2.is_finite() {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_INPUT,
            "bounds must be finite",
        ));
    }
    if x1 > x2 {
        return finalize_inconsistent_bounds(&options);
    }

    let outcome = run_solver(&function, x1, x2, &options, &mut cooperation)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    cooperation.completed(0).await?;
    finalize(outcome, &options)
}

fn parse_display(text: &str) -> BuiltinResult<OptimDisplay> {
    let display = OptimDisplay::parse(text)
        .filter(|display| runmat_builtins::OptimSolver::Fminbnd.accepts_display(*display))
        .ok_or_else(|| {
            fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!("option Display must be 'off', 'iter', 'notify', or 'final', got '{text}'"),
            )
        })?;
    Ok(if display == OptimDisplay::None {
        OptimDisplay::Off
    } else {
        display
    })
}

#[derive(Debug, Clone, Copy)]
struct FminbndOptions {
    tol_x: f64,
    max_iter: usize,
    max_fun_evals: usize,
    display: OptimDisplay,
}

impl FminbndOptions {
    fn from_struct(
        options: Option<&StructValue>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        crate::builtins::math::optim::common::ensure_selected_option_extensions(
            OptimSolver::Fminbnd,
            options,
            &FMINBND_OPTION_EXTENSION,
            &FMINBND_RESIDENT_EXTENSION,
            runtime,
        )?;
        let fields = ScalarOptionFields::from_struct(options);
        let display = match fields.get(OptimOptionField::Display) {
            Some(value) => parse_display(&option_string("Display", value)?)?,
            None => DEFAULT_DISPLAY,
        };
        let tol_x = match fields.get(OptimOptionField::TolX) {
            Some(value) => option_f64("TolX", value)?,
            None => DEFAULT_TOL_X,
        };
        if tol_x <= 0.0 {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                "option TolX must be positive",
            ));
        }
        let max_iter = match fields.get(OptimOptionField::MaxIter) {
            Some(value) => option_positive_usize("MaxIter", value)?,
            None => DEFAULT_MAX_ITER,
        };
        let max_fun_evals = match fields.get(OptimOptionField::MaxFunEvals) {
            Some(value) => option_positive_usize("MaxFunEvals", value)?,
            None => DEFAULT_MAX_FUN_EVALS,
        };
        Ok(Self {
            tol_x,
            max_iter,
            max_fun_evals,
            display,
        })
    }
}

fn parse_options(value: Option<Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options)),
        Some(_) => Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            "options must be a struct",
        )),
    }
}

fn option_string(field: &str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => {
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|error| {
                fminbnd_error_with_detail(
                    &FMINBND_ERROR_INVALID_ARGUMENT,
                    format!("option {field} contains {error}"),
                )
            })
        }
        _ => Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be a string"),
        )),
    }
}

fn option_f64(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) if crate::builtins::common::validation::integer_is_exact_f64(i) => i.to_f64(),
        Value::Int(_) => {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!("option {field} must be exactly representable as double"),
            ))
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if tensor.integer_storage().is_some_and(|storage| {
                storage
                    .exact_values()
                    .iter()
                    .any(|value| !crate::builtins::common::validation::integer_is_exact_f64(value))
            }) {
                return Err(fminbnd_error_with_detail(
                    &FMINBND_ERROR_INVALID_ARGUMENT,
                    format!("option {field} must be exactly representable as double"),
                ));
            }
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!("option {field} must be a real numeric scalar"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be finite"),
        ))
    }
}

fn option_positive_usize(field: &str, value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let Some(parsed) = integer.try_to_usize() else {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!("option {field} must be a positive integer"),
            ));
        };
        if parsed >= 1 {
            return Ok(parsed);
        }
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be a positive integer"),
        ));
    }
    let parsed = option_f64(field, value)?;
    if parsed < 1.0 {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be a positive integer"),
        ));
    }
    if parsed.fract() != 0.0 {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be an integer scalar"),
        ));
    }
    if parsed > usize::MAX as f64 || (usize::BITS == 64 && parsed == usize::MAX as f64) {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} exceeds maximum supported size"),
        ));
    }
    Ok(parsed as usize)
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(&t) => tensor::tensor_value_f64(&t, 0),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_INPUT,
                format!("{label} must be a finite real scalar"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_INPUT,
            format!("{label} must be finite"),
        ))
    }
}

#[derive(Debug, Clone)]
struct Outcome {
    inner: BrentMinResult,
}

async fn run_solver(
    function: &Value,
    lo: f64,
    hi: f64,
    options: &FminbndOptions,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let mut iter_log = IterDisplay::new(options.display);
    let observer: Option<&mut dyn BrentMinObserver> =
        if matches!(options.display, OptimDisplay::Iter) {
            Some(&mut iter_log)
        } else {
            None
        };
    let inner = brent_min(
        NAME,
        cooperation,
        function,
        lo,
        hi,
        BrentParams {
            tol_x: options.tol_x,
            max_iter: options.max_iter,
            max_fun_evals: options.max_fun_evals,
        },
        observer,
    )
    .await?;
    Ok(Outcome { inner })
}

fn finalize(outcome: Outcome, options: &FminbndOptions) -> BuiltinResult<Value> {
    let exit_flag = if outcome.inner.converged { 1 } else { 0 };
    let message = build_message(&outcome.inner);

    emit_summary(&outcome.inner, exit_flag, &message, options)?;

    let x = Value::Num(outcome.inner.x);
    let fval = Value::Num(outcome.inner.fval);
    let exitflag = Value::Num(exit_flag as f64);
    let output_struct = Value::Struct(build_output_struct(&outcome.inner, &message));

    match crate::output_count::current_output_count() {
        None => Ok(x),
        Some(0) => Ok(Value::output_list(Vec::new())),
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
        Some(_) => Ok(x),
    }
}

fn finalize_inconsistent_bounds(options: &FminbndOptions) -> BuiltinResult<Value> {
    let message = "Exiting: The bounds are inconsistent because x1 > x2.".to_string();
    emit_invalid_summary(-2, &message, options)?;

    let x = empty_double();
    let fval = empty_double();
    let exitflag = Value::Num(-2.0);
    let output_struct = Value::Struct(build_invalid_output_struct(&message));

    match crate::output_count::current_output_count() {
        None => Ok(x),
        Some(0) => Ok(Value::output_list(Vec::new())),
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
        Some(_) => Ok(empty_double()),
    }
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn build_output_struct(result: &BrentMinResult, message: &str) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("message", Value::from(message.to_string()));
    fields.finish()
}

fn build_invalid_output_struct(message: &str) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(0.0));
    fields.insert("funcCount", Value::Num(0.0));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("message", Value::from(message.to_string()));
    fields.finish()
}

fn build_message(result: &BrentMinResult) -> String {
    if result.converged {
        format!(
            "Optimization terminated: the current x satisfies the termination criteria using OPTIONS.TolX. Iterations: {}, FuncCount: {}.",
            result.iterations, result.func_count
        )
    } else {
        format!(
            "Exiting: Maximum number of function evaluations or iterations has been exceeded - increase MaxFunEvals or MaxIter. Iterations: {}, FuncCount: {}.",
            result.iterations, result.func_count
        )
    }
}

fn invalid_diagnostic_display() -> RuntimeError {
    let mut error = fminbnd_error_with_detail(
        &FMINBND_ERROR_INVALID_ARGUMENT,
        "diagnostic display is not supported by fminbnd",
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn emit_summary(
    result: &BrentMinResult,
    exit_flag: i32,
    message: &str,
    options: &FminbndOptions,
) -> BuiltinResult<()> {
    let should_emit = match options.display {
        OptimDisplay::Off | OptimDisplay::None => false,
        OptimDisplay::Final | OptimDisplay::Iter => true,
        OptimDisplay::Notify => exit_flag != 1,
        OptimDisplay::Diagnose => return Err(invalid_diagnostic_display()),
    };
    if !should_emit {
        return Ok(());
    }
    let line = format!(
        "fminbnd: x = {x:.6}, fval = {fval:.6}, exitflag = {exit_flag}. {message}",
        x = result.x,
        fval = result.fval,
    );
    crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line)
}

fn emit_invalid_summary(
    exit_flag: i32,
    message: &str,
    options: &FminbndOptions,
) -> BuiltinResult<()> {
    let should_emit = match options.display {
        OptimDisplay::Off | OptimDisplay::None => false,
        OptimDisplay::Final | OptimDisplay::Iter => true,
        OptimDisplay::Notify => exit_flag != 1,
        OptimDisplay::Diagnose => return Err(invalid_diagnostic_display()),
    };
    if should_emit {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("fminbnd: exitflag = {exit_flag}. {message}"),
        )?;
    }
    Ok(())
}

struct IterDisplay {
    mode: OptimDisplay,
    printed_header: bool,
}

impl IterDisplay {
    fn new(mode: OptimDisplay) -> Self {
        Self {
            mode,
            printed_header: false,
        }
    }
}

impl BrentMinObserver for IterDisplay {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentStepKind,
    ) -> BuiltinResult<()> {
        if !matches!(self.mode, OptimDisplay::Iter) {
            return Ok(());
        }
        if !self.printed_header {
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                " Func-count        x          f(x)          Procedure",
            )?;
            self.printed_header = true;
        }
        let procedure = match step_kind {
            BrentStepKind::Initial => "initial",
            BrentStepKind::GoldenSection => "golden",
            BrentStepKind::Parabolic => "parabolic",
        };
        let line =
            format!("    {func_count:>5}    {x:13.6e} {fx:13.6e}    {procedure}    (iter {iter})");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line)
    }
}

#[cfg(test)]
#[path = "fminbnd/tests.rs"]
mod tests;
