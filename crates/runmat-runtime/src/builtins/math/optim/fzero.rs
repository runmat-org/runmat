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
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::brent::{
    brent_zero, BrentParams, BrentZeroBracket, BrentZeroObserver, BrentZeroResult,
    BrentZeroStepKind,
};
use crate::builtins::math::optim::common::{
    call_scalar_function, option_f64, option_string, option_usize,
};
use crate::builtins::math::optim::type_resolvers::scalar_root_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fzero";
const ALGORITHM: &str = "bisection, interpolation";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS: usize = 500;

const FZERO_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Estimated root location.",
}];

const FZERO_OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated root location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function value at x.",
    },
];

const FZERO_OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated root location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function value at x.",
    },
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Convergence status code.",
    },
];

const FZERO_OUTPUT_ALL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated root location.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function value at x.",
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

const FZERO_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar-valued callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial point or two-element bracket.",
    },
];

const FZERO_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar-valued callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial point or two-element bracket.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimset.",
    },
];

const FZERO_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "x = fzero(fun, x0)",
        inputs: &FZERO_INPUTS_CORE,
        outputs: &FZERO_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = fzero(fun, x0, options)",
        inputs: &FZERO_INPUTS_WITH_OPTIONS,
        outputs: &FZERO_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fzero(fun, x0)",
        inputs: &FZERO_INPUTS_CORE,
        outputs: &FZERO_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fzero(fun, x0, options)",
        inputs: &FZERO_INPUTS_WITH_OPTIONS,
        outputs: &FZERO_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fzero(fun, x0)",
        inputs: &FZERO_INPUTS_CORE,
        outputs: &FZERO_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fzero(fun, x0, options)",
        inputs: &FZERO_INPUTS_WITH_OPTIONS,
        outputs: &FZERO_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fzero(fun, x0)",
        inputs: &FZERO_INPUTS_CORE,
        outputs: &FZERO_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fzero(fun, x0, options)",
        inputs: &FZERO_INPUTS_WITH_OPTIONS,
        outputs: &FZERO_OUTPUT_ALL,
    },
];

const FZERO_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FZERO.INVALID_ARGUMENT",
    identifier: Some("RunMat:fzero:InvalidArgument"),
    when: "Argument grammar/options struct are invalid.",
    message: "fzero: invalid argument",
};

const FZERO_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FZERO.INVALID_INPUT",
    identifier: Some("RunMat:fzero:InvalidInput"),
    when: "Callback/bracket/initial-point semantics are invalid.",
    message: "fzero: invalid input",
};

const FZERO_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FZERO.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fzero:TooManyOutputs"),
    when: "`fzero` is called with more than four requested output arguments.",
    message: "fzero: too many output arguments",
};

const FZERO_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FZERO_ERROR_INVALID_ARGUMENT,
    FZERO_ERROR_INVALID_INPUT,
    FZERO_ERROR_TOO_MANY_OUTPUTS,
];

pub const FZERO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FZERO_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FZERO_ERRORS,
};

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
        fzero_error_with_detail(fallback, err.message())
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
    category = "math/optim",
    summary = "Find scalar function zeros with bracketed root-finding.",
    keywords = "fzero,root finding,zero,brent,optimization",
    accel = "sink",
    type_resolver(scalar_root_type),
    descriptor(crate::builtins::math::optim::fzero::FZERO_DESCRIPTOR),
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
    let opts = FzeroOptions::from_struct(options.as_ref())
        .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_ARGUMENT))?;
    let bracket = initial_bracket(&function, x, &opts)
        .await
        .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_INPUT))?;
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
        observer,
    )
    .await
    .map_err(|err| fzero_map_error(err, &FZERO_ERROR_INVALID_INPUT))?;
    finalize(result, &opts)
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
    max_iter: usize,
    max_fun_evals: usize,
    display: DisplayMode,
}

impl FzeroOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
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
}

async fn initial_bracket(
    function: &Value,
    x: Value,
    options: &FzeroOptions,
) -> BuiltinResult<Bracket> {
    let x = crate::dispatcher::gather_if_needed_async(&x).await?;
    match x {
        Value::Tensor(tensor) if tensor.data.len() == 2 => {
            let a = tensor.data[0];
            let b = tensor.data[1];
            bracket_from_endpoints(function, a, b).await
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            expand_bracket(function, tensor.data[0], options).await
        }
        Value::Num(n) => expand_bracket(function, n, options).await,
        Value::Int(i) => expand_bracket(function, i.to_f64(), options).await,
        Value::Bool(b) => expand_bracket(function, if b { 1.0 } else { 0.0 }, options).await,
        other => Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            format!("initial point must be a scalar or two-element bracket, got {other:?}"),
        )),
    }
}

async fn bracket_from_endpoints(function: &Value, a: f64, b: f64) -> BuiltinResult<Bracket> {
    if !a.is_finite() || !b.is_finite() || a == b {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "bracket endpoints must be finite and distinct",
        ));
    }
    let fa = call_scalar_function(NAME, function, a).await?;
    if fa == 0.0 {
        return Ok(Bracket {
            a,
            b: a,
            fa,
            fb: fa,
            evals: 1,
        });
    }
    let fb = call_scalar_function(NAME, function, b).await?;
    if fb == 0.0 || fa.signum() != fb.signum() {
        Ok(Bracket {
            a,
            b,
            fa,
            fb,
            evals: 2,
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
) -> BuiltinResult<Bracket> {
    if !x0.is_finite() {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "initial point must be finite",
        ));
    }
    let f0 = call_scalar_function(NAME, function, x0).await?;
    if f0 == 0.0 {
        return Ok(Bracket {
            a: x0,
            b: x0,
            fa: f0,
            fb: f0,
            evals: 1,
        });
    }

    let mut evals = 1usize;
    let mut step = (x0.abs() * 0.01).max(0.01);
    while evals + 2 <= options.max_fun_evals {
        let a = x0 - step;
        let b = x0 + step;
        let fa = call_scalar_function(NAME, function, a).await?;
        let fb = call_scalar_function(NAME, function, b).await?;
        evals += 2;
        if fa == 0.0 {
            return Ok(Bracket {
                a,
                b: a,
                fa,
                fb: fa,
                evals,
            });
        }
        if fa.signum() != f0.signum() {
            return Ok(Bracket {
                a,
                b: x0,
                fa,
                fb: f0,
                evals,
            });
        }
        if fb.signum() != f0.signum() {
            return Ok(Bracket {
                a: x0,
                b,
                fa: f0,
                fb,
                evals,
            });
        }
        if fb == 0.0 || fa.signum() != fb.signum() {
            return Ok(Bracket {
                a,
                b,
                fa,
                fb,
                evals,
            });
        }
        step *= 1.6;
    }

    Err(fzero_error_with_detail(
        &FZERO_ERROR_INVALID_INPUT,
        "could not find a sign-changing bracket around the initial point",
    ))
}

fn finalize(result: BrentZeroResult, options: &FzeroOptions) -> BuiltinResult<Value> {
    let exit_flag = if result.converged { 1 } else { 0 };
    let message = build_message(&result);
    emit_summary(&result, exit_flag, &message, options);

    let x = Value::Num(result.x);
    let fval = Value::Num(result.fval);
    let exitflag = Value::Num(exit_flag as f64);
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
mod tests {
    use super::*;
    use crate::builtins::math::optim::brent::interpolation_step_accepted;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::Arc;

    #[test]
    fn fzero_bracketed_builtin_handle() {
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_scalar_initial_guess_expands_bracket() {
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("cos".into()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::FRAC_PI_2).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_scalar_initial_guess_uses_center_sign_for_bracket() {
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(std::f64::consts::FRAC_PI_2),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!(n.abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }
    #[test]
    fn fzero_accepts_semantic_function_handle_callback() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 42);
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar numeric argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x - 2.0)) })
            },
        )));

        let root = block_on(fzero_builtin(
            Value::BoundFunctionHandle {
                name: "root_function".to_string(),
                function: 42,
            },
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - 2.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_multi_output_two_returns_root_and_fval() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            Vec::new(),
        ))
        .expect("fzero");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                match (&outputs[0], &outputs[1]) {
                    (Value::Num(x), Value::Num(fval)) => {
                        assert!((x - std::f64::consts::PI).abs() < 1.0e-6);
                        assert!(fval.abs() < 1.0e-6);
                    }
                    other => panic!("unexpected outputs {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_multi_output_four_includes_output_struct() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            Vec::new(),
        ))
        .expect("fzero");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 4);
                assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
                match &outputs[3] {
                    Value::Struct(output) => {
                        assert!(matches!(
                            output.fields.get("iterations"),
                            Some(Value::Num(_))
                        ));
                        assert!(matches!(
                            output.fields.get("funcCount"),
                            Some(Value::Num(_))
                        ));
                        match output.fields.get("algorithm") {
                            Some(Value::String(text)) => assert!(text.contains("bisection")),
                            other => panic!("unexpected algorithm field {other:?}"),
                        }
                        match output.fields.get("message") {
                            Some(Value::String(text)) => assert!(text.contains("Zero found")),
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
    fn fzero_reports_zero_exitflag_when_iteration_budget_exhausted() {
        let mut opts = StructValue::new();
        opts.insert("MaxIter", Value::Num(1.0));
        opts.insert("MaxFunEvals", Value::Num(2.0));
        opts.insert("Display", Value::from("off"));
        let _guard = crate::output_count::push_output_count(Some(3));
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            vec![Value::Struct(opts)],
        ))
        .expect("fzero");
        match result {
            Value::OutputList(outputs) => match &outputs[2] {
                Value::Num(flag) => assert_eq!(*flag, 0.0),
                other => panic!("unexpected exitflag {other:?}"),
            },
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_reports_convergence_when_final_step_hits_root() {
        let mut opts = StructValue::new();
        opts.insert("MaxIter", Value::Num(1.0));
        opts.insert("Display", Value::from("off"));
        let _guard = crate::output_count::push_output_count(Some(3));
        let bracket = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            vec![Value::Struct(opts)],
        ))
        .expect("fzero");
        match result {
            Value::OutputList(outputs) => {
                assert!(matches!(&outputs[0], Value::Num(x) if x.abs() < 1.0e-12));
                assert!(matches!(&outputs[1], Value::Num(fval) if fval.abs() < 1.0e-12));
                assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_rejects_more_than_four_outputs() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            Vec::new(),
        ))
        .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), Some("RunMat:fzero:TooManyOutputs"));
        assert!(err.message().contains("maximum is 4"));
    }

    #[test]
    fn fzero_iter_display_records_iteration_rows() {
        crate::console::reset_thread_buffer();
        let mut opts = StructValue::new();
        opts.insert("Display", Value::from("iter"));
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            vec![Value::Struct(opts)],
        ))
        .expect("fzero");
        assert!(matches!(result, Value::Num(_)));

        let joined = crate::console::take_thread_buffer()
            .into_iter()
            .map(|entry| entry.text)
            .collect::<String>();
        assert!(joined.contains("Func-count"), "{joined}");
        assert!(joined.contains("initial"), "{joined}");
        assert!(
            joined.contains("interpolation") || joined.contains("bisection"),
            "{joined}"
        );
        assert!(joined.contains("exitflag = 1"), "{joined}");
    }

    #[test]
    fn brent_interpolation_acceptance_uses_signed_q() {
        assert!(!interpolation_step_accepted(1.0, -2.0, 1.0, 0.1, 10.0));
        assert!(interpolation_step_accepted(1.0, -2.0, -1.0, 0.1, 10.0));
    }

    #[test]
    fn fzero_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FZERO_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "x = fzero(fun, x0)",
                "x = fzero(fun, x0, options)",
                "[x, fval] = fzero(fun, x0)",
                "[x, fval] = fzero(fun, x0, options)",
                "[x, fval, exitflag] = fzero(fun, x0)",
                "[x, fval, exitflag] = fzero(fun, x0, options)",
                "[x, fval, exitflag, output] = fzero(fun, x0)",
                "[x, fval, exitflag, output] = fzero(fun, x0, options)",
            ]
        );

        let codes: Vec<&str> = FZERO_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.FZERO.INVALID_ARGUMENT",
                "RM.FZERO.INVALID_INPUT",
                "RM.FZERO.TOO_MANY_OUTPUTS",
            ]
        );
    }

    #[test]
    fn fzero_too_many_args_uses_stable_identifier() {
        let err = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            vec![
                Value::Struct(StructValue::new()),
                Value::Struct(StructValue::new()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:fzero:InvalidArgument"));
    }
}
