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
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::brent::{
    brent_min, BrentMinObserver, BrentMinResult, BrentParams, BrentStepKind,
};
use crate::builtins::math::optim::type_resolvers::scalar_root_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fminbnd";
const ALGORITHM: &str = "golden section search, parabolic interpolation";
const DEFAULT_TOL_X: f64 = 1.0e-4;
const DEFAULT_MAX_ITER: usize = 500;
const DEFAULT_MAX_FUN_EVALS: usize = 500;
const DEFAULT_DISPLAY: DisplayMode = DisplayMode::Notify;

const FMINBND_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Estimated minimizer location.",
}];

const FMINBND_OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated minimizer location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value at x.",
    },
];

const FMINBND_OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated minimizer location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value at x.",
    },
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Convergence status code.",
    },
];

const FMINBND_OUTPUT_ALL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated minimizer location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value at x.",
    },
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Convergence status code.",
    },
    BuiltinParamDescriptor {
        name: "output",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Iteration/function-count metadata struct.",
    },
];

const FMINBND_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar objective callback.",
    },
    BuiltinParamDescriptor {
        name: "x1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound.",
    },
    BuiltinParamDescriptor {
        name: "x2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound.",
    },
];

const FMINBND_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar objective callback.",
    },
    BuiltinParamDescriptor {
        name: "x1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound.",
    },
    BuiltinParamDescriptor {
        name: "x2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimset.",
    },
];

const FMINBND_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "x = fminbnd(fun, x1, x2)",
        inputs: &FMINBND_INPUTS_CORE,
        outputs: &FMINBND_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = fminbnd(fun, x1, x2, options)",
        inputs: &FMINBND_INPUTS_WITH_OPTIONS,
        outputs: &FMINBND_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fminbnd(fun, x1, x2)",
        inputs: &FMINBND_INPUTS_CORE,
        outputs: &FMINBND_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fminbnd(fun, x1, x2, options)",
        inputs: &FMINBND_INPUTS_WITH_OPTIONS,
        outputs: &FMINBND_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fminbnd(fun, x1, x2)",
        inputs: &FMINBND_INPUTS_CORE,
        outputs: &FMINBND_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fminbnd(fun, x1, x2, options)",
        inputs: &FMINBND_INPUTS_WITH_OPTIONS,
        outputs: &FMINBND_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminbnd(fun, x1, x2)",
        inputs: &FMINBND_INPUTS_CORE,
        outputs: &FMINBND_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminbnd(fun, x1, x2, options)",
        inputs: &FMINBND_INPUTS_WITH_OPTIONS,
        outputs: &FMINBND_OUTPUT_ALL,
    },
];

const FMINBND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINBND.INVALID_ARGUMENT",
    identifier: Some("RunMat:fminbnd:InvalidArgument"),
    when: "Argument grammar/options parsing is invalid.",
    message: "fminbnd: invalid argument",
};

const FMINBND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINBND.INVALID_INPUT",
    identifier: Some("RunMat:fminbnd:InvalidInput"),
    when: "Bounds/callback/input scalar semantics are invalid.",
    message: "fminbnd: invalid input",
};

const FMINBND_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FMINBND_ERROR_INVALID_ARGUMENT, FMINBND_ERROR_INVALID_INPUT];

pub const FMINBND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FMINBND_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FMINBND_ERRORS,
};

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
        fminbnd_error_with_detail(fallback, err.message())
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
    category = "math/optim",
    summary = "Find a local minimum of a scalar function on a bounded interval using Brent's method.",
    keywords = "fminbnd,bounded minimization,brent,golden section,parabolic interpolation,optimization",
    accel = "sink",
    type_resolver(scalar_root_type),
    descriptor(crate::builtins::math::optim::fminbnd::FMINBND_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fminbnd"
)]
async fn fminbnd_builtin(
    function: Value,
    x1: Value,
    x2: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options_struct = parse_options(rest.first())
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_ARGUMENT))?;
    let options = FminbndOptions::from_struct(options_struct.as_ref())
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_ARGUMENT))?;
    let x1 = scalar_bound("lower bound", x1)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    let x2 = scalar_bound("upper bound", x2)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;

    if !x1.is_finite() || !x2.is_finite() {
        return Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_INPUT,
            "bounds must be finite",
        ));
    }
    if x1 > x2 {
        return finalize_inconsistent_bounds(&options);
    }

    let outcome = run_solver(&function, x1, x2, &options)
        .await
        .map_err(|err| fminbnd_map_error(err, &FMINBND_ERROR_INVALID_INPUT))?;
    finalize(outcome, &options)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayMode {
    Off,
    Iter,
    Notify,
    Final,
}

impl DisplayMode {
    fn parse(text: &str) -> BuiltinResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(Self::Off),
            "iter" => Ok(Self::Iter),
            "notify" => Ok(Self::Notify),
            "final" => Ok(Self::Final),
            other => Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!(
                    "option Display must be 'off', 'iter', 'notify', or 'final', got '{other}'"
                ),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FminbndOptions {
    tol_x: f64,
    max_iter: usize,
    max_fun_evals: usize,
    display: DisplayMode,
}

impl FminbndOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        let display = match options {
            Some(opts) => match lookup(opts, "Display") {
                Some(value) => DisplayMode::parse(&option_string("Display", value)?)?,
                None => DEFAULT_DISPLAY,
            },
            None => DEFAULT_DISPLAY,
        };
        let tol_x = match options.and_then(|o| lookup(o, "TolX")) {
            Some(value) => option_f64("TolX", value)?,
            None => DEFAULT_TOL_X,
        };
        if tol_x <= 0.0 {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                "option TolX must be positive",
            ));
        }
        let max_iter = match options.and_then(|o| lookup(o, "MaxIter")) {
            Some(value) => option_positive_usize("MaxIter", value)?,
            None => DEFAULT_MAX_ITER,
        };
        let max_fun_evals = match options.and_then(|o| lookup(o, "MaxFunEvals")) {
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

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

fn lookup<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, v)| v)
}

fn option_string(field: &str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(fminbnd_error_with_detail(
            &FMINBND_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be a string, got {other:?}"),
        )),
    }
}

fn option_f64(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_ARGUMENT,
                format!("option {field} must be a real scalar, got {other:?}"),
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
    Ok(parsed as usize)
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
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
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(fminbnd_error_with_detail(
                &FMINBND_ERROR_INVALID_INPUT,
                format!("{label} must be a finite real scalar, got {other:?}"),
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
) -> BuiltinResult<Outcome> {
    let mut iter_log = IterDisplay::new(options.display);
    let observer: Option<&mut dyn BrentMinObserver> =
        if matches!(options.display, DisplayMode::Iter) {
            Some(&mut iter_log)
        } else {
            None
        };
    let inner = brent_min(
        NAME,
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

    emit_summary(&outcome.inner, exit_flag, &message, options);

    let x = Value::Num(outcome.inner.x);
    let fval = Value::Num(outcome.inner.fval);
    let exitflag = Value::Num(exit_flag as f64);
    let output_struct = Value::Struct(build_output_struct(&outcome.inner, &message));

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
        Some(n) if n >= 4 => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![x, fval, exitflag, output_struct],
        )),
        Some(_) => Ok(x),
    }
}

fn finalize_inconsistent_bounds(options: &FminbndOptions) -> BuiltinResult<Value> {
    let message = "Exiting: The bounds are inconsistent because x1 > x2.".to_string();
    emit_invalid_summary(-2, &message, options);

    let x = empty_double();
    let fval = empty_double();
    let exitflag = Value::Num(-2.0);
    let output_struct = Value::Struct(build_invalid_output_struct(&message));

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
        Some(n) if n >= 4 => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![x, fval, exitflag, output_struct],
        )),
        Some(_) => Ok(empty_double()),
    }
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn build_output_struct(result: &BrentMinResult, message: &str) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("message", Value::from(message.to_string()));
    fields
}

fn build_invalid_output_struct(message: &str) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(0.0));
    fields.insert("funcCount", Value::Num(0.0));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("message", Value::from(message.to_string()));
    fields
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

fn emit_summary(result: &BrentMinResult, exit_flag: i32, message: &str, options: &FminbndOptions) {
    let should_emit = match options.display {
        DisplayMode::Off => false,
        DisplayMode::Final | DisplayMode::Iter => true,
        DisplayMode::Notify => exit_flag != 1,
    };
    if !should_emit {
        return;
    }
    let line = format!(
        "fminbnd: x = {x:.6}, fval = {fval:.6}, exitflag = {exit_flag}. {message}",
        x = result.x,
        fval = result.fval,
    );
    crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line);
}

fn emit_invalid_summary(exit_flag: i32, message: &str, options: &FminbndOptions) {
    let should_emit = match options.display {
        DisplayMode::Off => false,
        DisplayMode::Final | DisplayMode::Iter => true,
        DisplayMode::Notify => exit_flag != 1,
    };
    if should_emit {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("fminbnd: exitflag = {exit_flag}. {message}"),
        );
    }
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

impl BrentMinObserver for IterDisplay {
    fn on_iteration(
        &mut self,
        iter: usize,
        func_count: usize,
        x: f64,
        fx: f64,
        step_kind: BrentStepKind,
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
            BrentStepKind::Initial => "initial",
            BrentStepKind::GoldenSection => "golden",
            BrentStepKind::Parabolic => "parabolic",
        };
        let line =
            format!("    {func_count:>5}    {x:13.6e} {fx:13.6e}    {procedure}    (iter {iter})");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::optim::brent::brent_min_tolerance;
    use futures::executor::block_on;
    use runmat_builtins::Value as V;

    const FMINBND_HELPER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "fx",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective scalar value.",
    }];

    const FMINBND_HELPER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar objective input.",
    }];

    const FMINBND_HELPER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "fx = __fminbnd_helper(x)",
            inputs: &FMINBND_HELPER_INPUTS,
            outputs: &FMINBND_HELPER_OUTPUT,
        }];

    const FMINBND_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];

    pub const FMINBND_TEST_HELPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &FMINBND_HELPER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &FMINBND_HELPER_ERRORS,
    };

    fn run_default(handle: &str, lo: f64, hi: f64) -> Value {
        block_on(fminbnd_builtin(
            V::FunctionHandle(handle.into()),
            V::Num(lo),
            V::Num(hi),
            Vec::new(),
        ))
        .expect("fminbnd")
    }

    fn run_with(handle: &str, lo: f64, hi: f64, extra: Vec<Value>) -> Value {
        block_on(fminbnd_builtin(
            V::FunctionHandle(handle.into()),
            V::Num(lo),
            V::Num(hi),
            extra,
        ))
        .expect("fminbnd")
    }

    #[test]
    fn fminbnd_test_helper_descriptor_is_attached_shape() {
        assert_eq!(
            FMINBND_TEST_HELPER_DESCRIPTOR.signatures[0].label,
            "fx = __fminbnd_helper(x)"
        );
    }

    #[runtime_builtin(
        name = "__fminbnd_quad_minus_two",
        type_resolver(crate::builtins::math::optim::type_resolvers::scalar_root_type),
        descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::fminbnd::tests"
    )]
    async fn quad_minus_two(x: Value) -> crate::BuiltinResult<Value> {
        let x = scalar_bound("x", x).await?;
        let diff = x - 2.0;
        Ok(Value::Num(diff * diff))
    }

    #[runtime_builtin(
        name = "__fminbnd_quad_minus_three",
        type_resolver(crate::builtins::math::optim::type_resolvers::scalar_root_type),
        descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::fminbnd::tests"
    )]
    async fn quad_minus_three(x: Value) -> crate::BuiltinResult<Value> {
        let x = scalar_bound("x", x).await?;
        let diff = x - 3.0;
        Ok(Value::Num(diff * diff))
    }

    #[runtime_builtin(
        name = "__fminbnd_multi_modal",
        type_resolver(crate::builtins::math::optim::type_resolvers::scalar_root_type),
        descriptor(crate::builtins::math::optim::fminbnd::tests::FMINBND_TEST_HELPER_DESCRIPTOR),
        builtin_path = "crate::builtins::math::optim::fminbnd::tests"
    )]
    async fn multi_modal(x: Value) -> crate::BuiltinResult<Value> {
        // 1 + sin(3x) on [0, 2π] — local minima near x ≈ π/2 + 2π/3 (etc.).
        let x = scalar_bound("x", x).await?;
        Ok(Value::Num(1.0 + (3.0 * x).sin()))
    }

    #[test]
    fn locates_smooth_quadratic_minimum() {
        let result = run_default("__fminbnd_quad_minus_two", 0.0, 5.0);
        match result {
            V::Num(x) => assert!((x - 2.0).abs() < 1.0e-3, "x = {x}"),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn locates_quadratic_minimum_offset_three() {
        let result = run_default("__fminbnd_quad_minus_three", 0.0, 5.0);
        match result {
            V::Num(x) => assert!((x - 3.0).abs() < 1.0e-3, "x = {x}"),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn locates_cosine_minimum_at_right_endpoint() {
        // cos(x) is monotonically decreasing on [0, π]; minimum is at x = π.
        let result = run_default("cos", 0.0, std::f64::consts::PI);
        match result {
            V::Num(x) => assert!((x - std::f64::consts::PI).abs() < 1.0e-3, "x = {x}"),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn returns_lone_endpoint_when_bounds_collapse() {
        let result = run_default("__fminbnd_quad_minus_two", 1.5, 1.5);
        match result {
            V::Num(x) => assert!((x - 1.5).abs() < 1.0e-12, "x = {x}"),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn reports_inconsistent_reversed_bounds() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let result = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(5.0),
            V::Num(0.0),
            Vec::new(),
        ))
        .expect("fminbnd");
        match result {
            V::OutputList(outputs) => {
                assert_eq!(outputs.len(), 4);
                assert!(matches!(&outputs[0], V::Tensor(t) if t.data.is_empty()));
                assert!(matches!(&outputs[1], V::Tensor(t) if t.data.is_empty()));
                assert!(matches!(&outputs[2], V::Num(flag) if *flag == -2.0));
                match &outputs[3] {
                    V::Struct(s) => {
                        assert!(matches!(s.fields.get("iterations"), Some(V::Num(0.0))));
                        assert!(matches!(s.fields.get("funcCount"), Some(V::Num(0.0))));
                        match s.fields.get("message") {
                            Some(V::String(text)) => assert!(text.contains("bounds")),
                            other => panic!("unexpected message field {other:?}"),
                        }
                    }
                    other => panic!("unexpected output struct {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn tolerance_is_additive_not_scaled_by_x() {
        let params = BrentParams {
            tol_x: 1.0e-4,
            max_iter: 500,
            max_fun_evals: 500,
        };
        let small = brent_min_tolerance(2.0, params);
        let large = brent_min_tolerance(1.0e9, params);
        assert!(small > params.tol_x);
        assert!(
            large < params.tol_x * 1.0e9,
            "large-scale tolerance was {large}"
        );
    }

    #[test]
    fn finds_local_minimum_in_multi_modal_function() {
        // 1 + sin(3x) on [1.5, 3.5] has its local minimum near x = π/2 ≈ 1.571 (sin(3π/2) = -1).
        let result = run_default("__fminbnd_multi_modal", 1.5, 3.5);
        match result {
            V::Num(x) => {
                let target = std::f64::consts::PI / 2.0;
                assert!((x - target).abs() < 5.0e-3, "x = {x}, target = {target}");
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn options_struct_overrides_default_tolerance() {
        let mut opts = StructValue::new();
        opts.insert("TolX", Value::Num(1.0e-12));
        let result = run_with(
            "__fminbnd_quad_minus_two",
            0.0,
            5.0,
            vec![Value::Struct(opts)],
        );
        match result {
            V::Num(x) => assert!((x - 2.0).abs() < 1.0e-6, "x = {x}"),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn max_fun_evals_default_is_independent_of_max_iter() {
        let mut opts = StructValue::new();
        opts.insert("MaxIter", Value::Num(1000.0));
        let parsed = FminbndOptions::from_struct(Some(&opts)).unwrap();
        assert_eq!(parsed.max_iter, 1000);
        assert_eq!(parsed.max_fun_evals, DEFAULT_MAX_FUN_EVALS);
    }

    #[test]
    fn rejects_nonfinite_bounds() {
        let err = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(f64::NAN),
            V::Num(5.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().to_ascii_lowercase().contains("finite"));
    }

    #[test]
    fn rejects_invalid_options_type() {
        let err = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            vec![Value::Num(1.0)],
        ))
        .unwrap_err();
        assert!(err.message().to_ascii_lowercase().contains("options"));
    }

    #[test]
    fn rejects_nonpositive_tol_x() {
        let mut opts = StructValue::new();
        opts.insert("TolX", Value::Num(0.0));
        let err = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            vec![Value::Struct(opts)],
        ))
        .unwrap_err();
        assert!(err.message().to_lowercase().contains("tolx"));
    }

    #[test]
    fn rejects_unknown_display_value() {
        let mut opts = StructValue::new();
        opts.insert("Display", Value::from("loud"));
        let err = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            vec![Value::Struct(opts)],
        ))
        .unwrap_err();
        assert!(err.message().to_lowercase().contains("display"));
    }

    #[test]
    fn multi_output_two_returns_x_and_fval() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            Vec::new(),
        ))
        .expect("fminbnd");
        match result {
            V::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                match (&outputs[0], &outputs[1]) {
                    (V::Num(x), V::Num(fval)) => {
                        assert!((x - 2.0).abs() < 1.0e-3);
                        assert!(fval.abs() < 1.0e-5);
                    }
                    other => panic!("unexpected outputs {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn multi_output_three_includes_exitflag() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            Vec::new(),
        ))
        .expect("fminbnd");
        match result {
            V::OutputList(outputs) => {
                assert_eq!(outputs.len(), 3);
                match &outputs[2] {
                    V::Num(flag) => assert!((*flag - 1.0).abs() < 1.0e-12),
                    other => panic!("unexpected exitflag {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn multi_output_four_includes_output_struct() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let result = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            Vec::new(),
        ))
        .expect("fminbnd");
        match result {
            V::OutputList(outputs) => {
                assert_eq!(outputs.len(), 4);
                match &outputs[3] {
                    V::Struct(s) => {
                        assert!(matches!(s.fields.get("iterations"), Some(V::Num(_))));
                        assert!(matches!(s.fields.get("funcCount"), Some(V::Num(_))));
                        match s.fields.get("algorithm") {
                            Some(V::String(text)) => assert!(text.contains("golden")),
                            other => panic!("unexpected algorithm field {other:?}"),
                        }
                        assert!(s.fields.get("message").is_some());
                    }
                    other => panic!("unexpected output struct {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn reports_zero_exitflag_when_max_iter_exhausted() {
        let mut opts = StructValue::new();
        opts.insert("MaxIter", Value::Num(1.0));
        opts.insert("MaxFunEvals", Value::Num(2.0));
        opts.insert("Display", Value::from("off"));
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            vec![Value::Struct(opts)],
        ))
        .expect("fminbnd");
        match result {
            V::OutputList(outputs) => match &outputs[2] {
                V::Num(flag) => assert_eq!(*flag, 0.0),
                other => panic!("unexpected exitflag {other:?}"),
            },
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fminbnd_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FMINBND_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "x = fminbnd(fun, x1, x2)",
                "x = fminbnd(fun, x1, x2, options)",
                "[x, fval] = fminbnd(fun, x1, x2)",
                "[x, fval] = fminbnd(fun, x1, x2, options)",
                "[x, fval, exitflag] = fminbnd(fun, x1, x2)",
                "[x, fval, exitflag] = fminbnd(fun, x1, x2, options)",
                "[x, fval, exitflag, output] = fminbnd(fun, x1, x2)",
                "[x, fval, exitflag, output] = fminbnd(fun, x1, x2, options)",
            ]
        );

        let codes: Vec<&str> = FMINBND_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.FMINBND.INVALID_ARGUMENT", "RM.FMINBND.INVALID_INPUT"]
        );
    }

    #[test]
    fn fminbnd_too_many_args_uses_stable_identifier() {
        let err = block_on(fminbnd_builtin(
            V::FunctionHandle("__fminbnd_quad_minus_two".into()),
            V::Num(0.0),
            V::Num(5.0),
            vec![
                Value::Struct(StructValue::new()),
                Value::Struct(StructValue::new()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:fminbnd:InvalidArgument"));
    }
}
