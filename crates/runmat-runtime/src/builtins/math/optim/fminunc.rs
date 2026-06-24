//! MATLAB-compatible `fminunc` builtin for unconstrained smooth minimization.

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
use crate::builtins::math::optim::common::{
    call_function, initial_guess, lookup_option, option_f64, option_string, value_to_real_vector,
    value_to_scalar, vector_to_value,
};
use crate::builtins::math::optim::type_resolvers::nonlinear_solve_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fminunc";
const ALGORITHM: &str = "quasi-newton bfgs";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS: usize = 40000;
const MAX_ITER_LIMIT: usize = 1_000_000;
const MAX_FUN_EVAL_LIMIT: usize = 10_000_000;
const MAX_DENSE_BFGS_VARIABLES: usize = 2048;
const C1: f64 = 1.0e-4;
const C2: f64 = 0.9;
const MAX_LINE_SEARCH_ITERS: usize = 24;

macro_rules! output_x {
    () => {
        BuiltinParamDescriptor {
            name: "x",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Estimated local minimizer.",
        }
    };
}

macro_rules! output_fval {
    () => {
        BuiltinParamDescriptor {
            name: "fval",
            ty: BuiltinParamType::NumericScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Objective value at x.",
        }
    };
}

macro_rules! output_exitflag {
    () => {
        BuiltinParamDescriptor {
            name: "exitflag",
            ty: BuiltinParamType::NumericScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Convergence status code.",
        }
    };
}

macro_rules! output_output {
    () => {
        BuiltinParamDescriptor {
            name: "output",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Iteration/function-count metadata struct.",
        }
    };
}

macro_rules! output_grad {
    () => {
        BuiltinParamDescriptor {
            name: "grad",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Gradient at x.",
        }
    };
}

macro_rules! output_hessian {
    () => {
        BuiltinParamDescriptor {
            name: "hessian",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Final approximate Hessian matrix.",
        }
    };
}

const FMINUNC_OUTPUT_X: [BuiltinParamDescriptor; 1] = [output_x!()];

const FMINUNC_OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [output_x!(), output_fval!()];

const FMINUNC_OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] =
    [output_x!(), output_fval!(), output_exitflag!()];

const FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT: [BuiltinParamDescriptor; 4] = [
    output_x!(),
    output_fval!(),
    output_exitflag!(),
    output_output!(),
];

const FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT_GRAD: [BuiltinParamDescriptor; 5] = [
    output_x!(),
    output_fval!(),
    output_exitflag!(),
    output_output!(),
    output_grad!(),
];

const FMINUNC_OUTPUT_ALL: [BuiltinParamDescriptor; 6] = [
    output_x!(),
    output_fval!(),
    output_exitflag!(),
    output_output!(),
    output_grad!(),
    output_hessian!(),
];

macro_rules! input_fun {
    () => {
        BuiltinParamDescriptor {
            name: "fun",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Scalar objective callback.",
        }
    };
}

macro_rules! input_x0 {
    () => {
        BuiltinParamDescriptor {
            name: "x0",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Initial guess scalar/vector/array.",
        }
    };
}

const FMINUNC_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [input_fun!(), input_x0!()];

const FMINUNC_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    input_fun!(),
    input_x0!(),
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimoptions or optimset.",
    },
];

const FMINUNC_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "x = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, grad] = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT_GRAD,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, grad] = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_X_FVAL_EXITFLAG_OUTPUT_GRAD,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, grad, hessian] = fminunc(fun, x0)",
        inputs: &FMINUNC_INPUTS_CORE,
        outputs: &FMINUNC_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, grad, hessian] = fminunc(fun, x0, options)",
        inputs: &FMINUNC_INPUTS_WITH_OPTIONS,
        outputs: &FMINUNC_OUTPUT_ALL,
    },
];

const FMINUNC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINUNC.INVALID_ARGUMENT",
    identifier: Some("RunMat:fminunc:InvalidArgument"),
    when: "Argument grammar/options parsing is invalid.",
    message: "fminunc: invalid argument",
};

const FMINUNC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINUNC.INVALID_INPUT",
    identifier: Some("RunMat:fminunc:InvalidInput"),
    when: "Initial guess/callback/iteration semantics are invalid.",
    message: "fminunc: invalid input",
};

const FMINUNC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FMINUNC_ERROR_INVALID_ARGUMENT, FMINUNC_ERROR_INVALID_INPUT];

pub const FMINUNC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FMINUNC_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FMINUNC_ERRORS,
};

fn fminunc_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fminunc:") {
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

fn fminunc_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        fminunc_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fminunc")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fminunc",
    op_kind: GpuOpKind::Custom("unconstrained-min"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host BFGS solver with strong-Wolfe line search. Callback computations may use GPU-aware builtins, but optimizer state is gathered on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fminunc")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fminunc",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Unconstrained minimization repeatedly invokes user callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fminunc",
    category = "math/optim",
    summary = "Find an unconstrained local minimum of a smooth scalar objective.",
    keywords = "fminunc,unconstrained minimization,bfgs,quasi-newton,strong wolfe,optimization",
    accel = "sink",
    type_resolver(nonlinear_solve_type),
    descriptor(crate::builtins::math::optim::fminunc::FMINUNC_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fminunc"
)]
async fn fminunc_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options_struct = parse_options(rest.first())
        .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let guess = initial_guess(NAME, x0)
        .await
        .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_INPUT))?;
    let options = FminuncOptions::from_struct(options_struct.as_ref(), guess.values.len())
        .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_ARGUMENT))?;
    let outcome = minimize(
        &function,
        guess.values,
        &guess.shape,
        guess.scalar,
        &options,
    )
    .await
    .map_err(|err| fminunc_map_error(err, &FMINUNC_ERROR_INVALID_INPUT))?;
    finalize(outcome, &guess.shape, guess.scalar, &options)
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
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
            other => Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                format!(
                    "option Display must be 'off', 'iter', 'notify', or 'final', got '{other}'"
                ),
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct FminuncOptions {
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: usize,
    display: DisplayMode,
    specify_objective_gradient: bool,
}

impl FminuncOptions {
    fn from_struct(options: Option<&StructValue>, variables: usize) -> BuiltinResult<Self> {
        let display = DisplayMode::parse(&option_string(options, "Display", "off")?)?;
        let algorithm = option_string(options, "Algorithm", "quasi-newton")?;
        if !matches!(algorithm.as_str(), "quasi-newton" | "bfgs") {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "fminunc: only Algorithm='quasi-newton' is supported",
            ));
        }
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        let tol_fun = option_f64(NAME, options, "TolFun", DEFAULT_TOL_FUN)?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter =
            bounded_option_usize(options, "MaxIter", DEFAULT_MAX_ITER, MAX_ITER_LIMIT)?.max(1);
        let max_fun_evals = bounded_option_usize(
            options,
            "MaxFunEvals",
            DEFAULT_MAX_FUN_EVALS,
            MAX_FUN_EVAL_LIMIT,
        )?
        .max(1);
        let specify_objective_gradient = option_bool(options, "SpecifyObjectiveGradient", false)?;
        validate_problem_size(variables)?;
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
            display,
            specify_objective_gradient,
        })
    }
}

fn bounded_option_usize(
    options: Option<&StructValue>,
    field: &str,
    default: usize,
    maximum: usize,
) -> BuiltinResult<usize> {
    let value = option_f64(NAME, options, field, default as f64)?;
    if value < 0.0 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be non-negative"),
        ));
    }
    if value > maximum as f64 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be no greater than {maximum}"),
        ));
    }
    if value.fract() != 0.0 {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be an integer"),
        ));
    }
    Ok(value.floor() as usize)
}

fn validate_problem_size(variables: usize) -> BuiltinResult<()> {
    if variables > MAX_DENSE_BFGS_VARIABLES {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "problem has {variables} variables; current dense BFGS implementation supports at most {MAX_DENSE_BFGS_VARIABLES}"
            ),
        ));
    }
    dense_len(variables)?;
    Ok(())
}

fn option_bool(options: Option<&StructValue>, field: &str, default: bool) -> BuiltinResult<bool> {
    let Some(options) = options else {
        return Ok(default);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    bool_value(field, value)
}

fn bool_value(field: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) => bool_from_number(field, *n),
        Value::Int(i) => bool_from_number(field, i.to_f64()),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => bool_from_number(field, data[0]),
        Value::String(s) => bool_from_text(field, s),
        Value::StringArray(sa) if sa.data.len() == 1 => bool_from_text(field, &sa.data[0]),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            bool_from_text(field, &text)
        }
        other => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be logical, got {other:?}"),
        )),
    }
}

fn bool_from_number(field: &str, value: f64) -> BuiltinResult<bool> {
    if !value.is_finite() {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be finite"),
        ));
    }
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be logical 0 or 1"),
        ))
    }
}

fn bool_from_text(field: &str, value: &str) -> BuiltinResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" => Ok(true),
        "off" | "false" | "no" => Ok(false),
        other => Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            format!("option {field} must be 'on' or 'off', got '{other}'"),
        )),
    }
}

#[derive(Debug, Clone)]
struct ObjectiveState {
    x: Vec<f64>,
    f: f64,
    grad: Vec<f64>,
}

#[derive(Debug, Clone)]
struct MinimizeOutcome {
    x: Vec<f64>,
    fval: f64,
    grad: Vec<f64>,
    inverse_hessian: Vec<f64>,
    iterations: usize,
    func_count: usize,
    exitflag: i32,
    firstorderopt: f64,
    stepsize: f64,
    message: String,
}

struct ObjectiveEvaluator<'a> {
    function: &'a Value,
    shape: Vec<usize>,
    scalar: bool,
    specify_gradient: bool,
    max_fun_evals: usize,
    func_count: usize,
}

impl<'a> ObjectiveEvaluator<'a> {
    fn new(
        function: &'a Value,
        shape: &[usize],
        scalar: bool,
        specify_gradient: bool,
        max_fun_evals: usize,
    ) -> Self {
        Self {
            function,
            shape: shape.to_vec(),
            scalar,
            specify_gradient,
            max_fun_evals,
            func_count: 0,
        }
    }

    async fn evaluate(&mut self, x: &[f64]) -> BuiltinResult<ObjectiveState> {
        self.ensure_state_budget(x.len())?;
        if self.specify_gradient {
            self.evaluate_with_gradient(x).await
        } else {
            let f = self.evaluate_value(x).await?;
            let grad = self.forward_difference_gradient(x, f).await?;
            Ok(ObjectiveState {
                x: x.to_vec(),
                f,
                grad,
            })
        }
    }

    async fn evaluate_value(&mut self, x: &[f64]) -> BuiltinResult<f64> {
        self.ensure_value_budget()?;
        let arg = vector_to_value(NAME, x.to_vec(), &self.shape, self.scalar)?;
        let value = call_function(self.function, vec![arg]).await?;
        let value = crate::dispatcher::gather_if_needed_async(&value).await?;
        self.func_count += 1;
        value_to_scalar(NAME, value)
    }

    async fn evaluate_with_gradient(&mut self, x: &[f64]) -> BuiltinResult<ObjectiveState> {
        let callback =
            crate::canonicalize_callback_handle_for_semantic_resolution(self.function.clone());
        let arg = vector_to_value(NAME, x.to_vec(), &self.shape, self.scalar)?;
        let value = crate::call_feval_async_with_outputs(callback, &[arg], 2).await?;
        let value = crate::dispatcher::gather_if_needed_async(&value).await?;
        self.func_count += 1;
        let Value::OutputList(outputs) = value else {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_INPUT,
                "objective must return [f, g] when SpecifyObjectiveGradient is true",
            ));
        };
        if outputs.len() < 2 {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_INPUT,
                "objective must return both objective value and gradient",
            ));
        }
        let f = value_to_scalar(NAME, outputs[0].clone())?;
        let grad = value_to_real_vector(NAME, outputs[1].clone()).await?;
        if grad.len() != x.len() {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_INPUT,
                format!(
                    "objective gradient length {} does not match x length {}",
                    grad.len(),
                    x.len()
                ),
            ));
        }
        Ok(ObjectiveState {
            x: x.to_vec(),
            f,
            grad,
        })
    }

    async fn forward_difference_gradient(&mut self, x: &[f64], f0: f64) -> BuiltinResult<Vec<f64>> {
        let mut grad = vec![0.0; x.len()];
        for i in 0..x.len() {
            let step = finite_difference_step(x[i]);
            let mut trial = x.to_vec();
            trial[i] += step;
            let f_step = self.evaluate_value(&trial).await?;
            grad[i] = (f_step - f0) / step;
        }
        Ok(grad)
    }

    fn state_evaluation_cost(&self, variables: usize) -> Option<usize> {
        if self.specify_gradient {
            Some(1)
        } else {
            variables.checked_add(1)
        }
    }

    fn remaining_func_evals(&self) -> usize {
        self.max_fun_evals.saturating_sub(self.func_count)
    }

    fn can_evaluate_state(&self, variables: usize) -> bool {
        self.state_evaluation_cost(variables)
            .is_some_and(|cost| cost <= self.remaining_func_evals())
    }

    fn can_evaluate_value(&self) -> bool {
        self.remaining_func_evals() >= 1
    }

    fn ensure_state_budget(&self, variables: usize) -> BuiltinResult<()> {
        if self.can_evaluate_state(variables) {
            return Ok(());
        }
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "MaxFunEvals exceeded before completing objective/gradient evaluation (FuncCount {}, MaxFunEvals {})",
                self.func_count, self.max_fun_evals
            ),
        ))
    }

    fn ensure_value_budget(&self) -> BuiltinResult<()> {
        if self.can_evaluate_value() {
            return Ok(());
        }
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "MaxFunEvals exceeded before objective evaluation (FuncCount {}, MaxFunEvals {})",
                self.func_count, self.max_fun_evals
            ),
        ))
    }
}

fn finite_difference_step(x: f64) -> f64 {
    f64::EPSILON.sqrt() * x.abs().max(1.0)
}

async fn minimize(
    function: &Value,
    x0: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FminuncOptions,
) -> BuiltinResult<MinimizeOutcome> {
    let n = x0.len();
    validate_problem_size(n)?;
    let mut evaluator = ObjectiveEvaluator::new(
        function,
        shape,
        scalar,
        options.specify_objective_gradient,
        options.max_fun_evals,
    );
    let mut state = evaluator.evaluate(&x0).await?;
    let mut inverse_hessian = identity(n)?;
    let mut iterations = 0usize;
    let mut stepsize = 0.0f64;
    let mut exitflag = 0i32;
    let mut line_search_failed = false;

    if norm_inf(&state.grad) <= options.tol_fun {
        exitflag = 1;
    }

    while exitflag == 0
        && iterations < options.max_iter
        && evaluator.func_count < options.max_fun_evals
    {
        let mut direction = mat_vec(&inverse_hessian, &state.grad, n);
        for value in &mut direction {
            *value = -*value;
        }
        if dot(&direction, &state.grad) >= -1.0e-14 || !all_finite(&direction) {
            inverse_hessian = identity(n)?;
            direction = state.grad.iter().map(|value| -value).collect();
        }

        let next = match line_search(&mut evaluator, &state, &direction).await? {
            LineSearchResult::Step(next) => next,
            LineSearchResult::BudgetExhausted => {
                exitflag = 0;
                break;
            }
            LineSearchResult::Failed => {
                line_search_failed = true;
                exitflag = -3;
                break;
            }
        };

        let s = subtract(&next.x, &state.x);
        stepsize = norm_inf(&s);
        let y = subtract(&next.grad, &state.grad);
        update_inverse_hessian(&mut inverse_hessian, &s, &y, n);
        let f_delta = (state.f - next.f).abs();
        let x_scale = 1.0 + norm_inf(&state.x);
        let f_scale = 1.0 + state.f.abs();
        state = next;
        iterations += 1;

        if norm_inf(&state.grad) <= options.tol_fun {
            exitflag = 1;
        } else if stepsize <= options.tol_x * x_scale {
            exitflag = 2;
        } else if f_delta <= options.tol_fun * f_scale {
            exitflag = 3;
        }
    }

    let firstorderopt = norm_inf(&state.grad);
    let message = build_message(
        exitflag,
        iterations,
        evaluator.func_count,
        line_search_failed,
    );
    emit_summary(
        &state,
        exitflag,
        &message,
        options,
        iterations,
        evaluator.func_count,
    );

    Ok(MinimizeOutcome {
        x: state.x,
        fval: state.f,
        grad: state.grad,
        inverse_hessian,
        iterations,
        func_count: evaluator.func_count,
        exitflag,
        firstorderopt,
        stepsize,
        message,
    })
}

async fn line_search(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = dot(&state.grad, direction);
    if dphi0 >= 0.0 || !dphi0.is_finite() {
        return Ok(LineSearchResult::Failed);
    }

    let mut alpha_prev = 0.0;
    let mut f_prev = state.f;
    let mut alpha = 1.0;
    for iter in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let trial_x = add_scaled(&state.x, direction, alpha);
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f > state.f + C1 * alpha * dphi0 || (iter > 0 && trial.f >= f_prev) {
            return zoom(evaluator, state, direction, alpha_prev, alpha).await;
        }
        let dphi = dot(&trial.grad, direction);
        if dphi.abs() <= -C2 * dphi0 {
            return Ok(LineSearchResult::Step(trial));
        }
        if dphi >= 0.0 {
            return zoom(evaluator, state, direction, alpha, alpha_prev).await;
        }
        alpha_prev = alpha;
        f_prev = trial.f;
        alpha = (alpha * 2.0).min(64.0);
    }
    Ok(LineSearchResult::Failed)
}

enum LineSearchResult {
    Step(ObjectiveState),
    BudgetExhausted,
    Failed,
}

async fn zoom(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
    mut lo: f64,
    mut hi: f64,
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = dot(&state.grad, direction);
    let mut f_lo = if lo == 0.0 {
        state.f
    } else {
        if !evaluator.can_evaluate_value() {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let x_lo = add_scaled(&state.x, direction, lo);
        evaluator.evaluate_value(&x_lo).await?
    };
    for _ in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let alpha = 0.5 * (lo + hi);
        let trial_x = add_scaled(&state.x, direction, alpha);
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f > state.f + C1 * alpha * dphi0 || trial.f >= f_lo {
            hi = alpha;
        } else {
            let dphi = dot(&trial.grad, direction);
            if dphi.abs() <= -C2 * dphi0 {
                return Ok(LineSearchResult::Step(trial));
            }
            if dphi * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            f_lo = trial.f;
        }
        if (hi - lo).abs() <= 1.0e-12 * (1.0 + lo.abs() + hi.abs()) {
            if trial.f <= state.f + C1 * alpha * dphi0 {
                return Ok(LineSearchResult::Step(trial));
            }
            return Ok(LineSearchResult::Failed);
        }
    }
    Ok(LineSearchResult::Failed)
}

fn update_inverse_hessian(h: &mut [f64], s: &[f64], y: &[f64], n: usize) {
    let ys = dot(y, s);
    if ys <= 1.0e-14 || !ys.is_finite() {
        return;
    }
    let rho = 1.0 / ys;
    let hy = mat_vec(h, y, n);
    let yhy = dot(y, &hy);
    let coeff = (1.0 + yhy * rho) * rho;
    for row in 0..n {
        for col in 0..n {
            h[row * n + col] +=
                coeff * s[row] * s[col] - rho * (s[row] * hy[col] + hy[row] * s[col]);
        }
    }
}

fn finalize(
    outcome: MinimizeOutcome,
    shape: &[usize],
    scalar: bool,
    _options: &FminuncOptions,
) -> BuiltinResult<Value> {
    let x = vector_to_value(NAME, outcome.x.clone(), shape, scalar)?;
    let fval = Value::Num(outcome.fval);
    let exitflag = Value::Num(outcome.exitflag as f64);
    let output = Value::Struct(build_output_struct(&outcome));
    let grad = vector_to_value(NAME, outcome.grad.clone(), shape, scalar)?;
    let hessian = hessian_value(&outcome.inverse_hessian, outcome.grad.len(), scalar)?;

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
            vec![x, fval, exitflag, output],
        )),
        Some(5) => Ok(crate::output_count::output_list_with_padding(
            5,
            vec![x, fval, exitflag, output, grad],
        )),
        Some(n) if n >= 6 => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![x, fval, exitflag, output, grad, hessian],
        )),
        Some(_) => unreachable!("output count cases above are exhaustive"),
    }
}

fn hessian_value(inverse_hessian: &[f64], n: usize, scalar: bool) -> BuiltinResult<Value> {
    let hessian = match invert_matrix(inverse_hessian, n) {
        Some(hessian) => hessian,
        None => identity(n)?,
    };
    if scalar {
        Ok(Value::Num(hessian[0]))
    } else {
        Tensor::new(hessian, vec![n, n])
            .map(Value::Tensor)
            .map_err(|e| fminunc_error_with_detail(&FMINUNC_ERROR_INVALID_INPUT, e))
    }
}

fn build_output_struct(outcome: &MinimizeOutcome) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("funcCount", Value::Num(outcome.func_count as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("firstorderopt", Value::Num(outcome.firstorderopt));
    fields.insert("stepsize", Value::Num(outcome.stepsize));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields
}

fn build_message(
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    line_search_failed: bool,
) -> String {
    match exitflag {
        1 => format!(
            "Optimization terminated: first-order optimality is below OPTIONS.TolFun. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        2 => format!(
            "Optimization terminated: step size is below OPTIONS.TolX. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        3 => format!(
            "Optimization terminated: objective change is below OPTIONS.TolFun. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        -3 if line_search_failed => format!(
            "Exiting: line search could not find a point satisfying the strong Wolfe conditions. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        _ => format!(
            "Exiting: Maximum number of function evaluations or iterations has been exceeded. Iterations: {iterations}, FuncCount: {func_count}."
        ),
    }
}

fn emit_summary(
    state: &ObjectiveState,
    exitflag: i32,
    message: &str,
    options: &FminuncOptions,
    iterations: usize,
    func_count: usize,
) {
    let should_emit = match options.display {
        DisplayMode::Off => false,
        DisplayMode::Final | DisplayMode::Iter => true,
        DisplayMode::Notify => exitflag <= 0,
    };
    if !should_emit {
        return;
    }
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "fminunc: fval = {fval:.6e}, firstorderopt = {opt:.6e}, exitflag = {exitflag}. {message}",
            fval = state.f,
            opt = norm_inf(&state.grad),
        ),
    );
    if matches!(options.display, DisplayMode::Iter) {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("fminunc: iterations = {iterations}, funcCount = {func_count}"),
        );
    }
}

fn dense_len(n: usize) -> BuiltinResult<usize> {
    n.checked_mul(n).ok_or_else(|| {
        fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            "dense BFGS matrix size overflows usize",
        )
    })
}

fn identity(n: usize) -> BuiltinResult<Vec<f64>> {
    let len = dense_len(n)?;
    let mut out = vec![0.0; len];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    Ok(out)
}

fn invert_matrix(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = matrix.to_vec();
    let mut inv = identity(n).ok()?;
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = a[row * n + col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs <= 1.0e-14 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
                inv.swap(col * n + j, pivot * n + j);
            }
        }
        let diag = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= diag;
            inv[col * n + j] /= diag;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[row * n + j] -= factor * a[col * n + j];
                inv[row * n + j] -= factor * inv[col * n + j];
            }
        }
    }
    Some(inv)
}

fn mat_vec(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for row in 0..n {
        for col in 0..n {
            out[row] += matrix[row * n + col] * vector[col];
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn add_scaled(x: &[f64], direction: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(direction.iter())
        .map(|(xi, di)| xi + alpha * di)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;
    use std::sync::Arc;

    fn bound(id: usize) -> Value {
        Value::BoundFunctionHandle {
            name: format!("objective_{id}"),
            function: id,
        }
    }

    fn tensor(data: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(data, vec![3, 1]).unwrap())
    }

    fn vector_from_value(value: &Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data.clone(),
            Value::Num(n) => vec![*n],
            other => panic!("expected numeric value, got {other:?}"),
        }
    }

    fn assert_close_vec(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= tol, "at {idx}: expected {e}, got {a}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_quadratic_reaches_vector_minimum() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = vector_from_value(&args[0]);
                Box::pin(async move {
                    let target = [1.0, 2.0, 3.0];
                    let f = x
                        .iter()
                        .zip(target.iter())
                        .map(|(xi, ti)| (xi - ti).powi(2))
                        .sum::<f64>();
                    Ok(Value::Num(f))
                })
            },
        )));
        let result = block_on(fminunc_builtin(
            bound(101),
            tensor(vec![0.0, 0.0, 0.0]),
            Vec::new(),
        ))
        .expect("fminunc");
        assert_close_vec(&vector_from_value(&result), &[1.0, 2.0, 3.0], 1.0e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rosenbrock_converges_from_standard_start() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = vector_from_value(&args[0]);
                Box::pin(async move {
                    let a = x[0];
                    let b = x[1];
                    Ok(Value::Num(100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2)))
                })
            },
        )));
        let x0 = Value::Tensor(Tensor::new(vec![-1.2, 1.0], vec![2, 1]).unwrap());
        let result = block_on(fminunc_builtin(bound(102), x0, Vec::new())).expect("fminunc");
        assert_close_vec(&vector_from_value(&result), &[1.0, 1.0], 2.0e-3);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_handles_high_dimensional_smooth_objective() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = vector_from_value(&args[0]);
                Box::pin(async move {
                    let f = x
                        .iter()
                        .enumerate()
                        .map(|(idx, xi)| {
                            let target = (idx + 1) as f64;
                            (xi - target).powi(2)
                        })
                        .sum::<f64>();
                    Ok(Value::Num(f))
                })
            },
        )));
        let x0 = Value::Tensor(Tensor::new(vec![0.0; 8], vec![8, 1]).unwrap());
        let result = block_on(fminunc_builtin(bound(103), x0, Vec::new())).expect("fminunc");
        assert_close_vec(
            &vector_from_value(&result),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            1.0e-4,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_uses_objective_gradient_and_returns_six_outputs() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 2);
                let x = vector_from_value(&args[0]);
                Box::pin(async move {
                    let target = [1.0, 2.0, 3.0];
                    let grad = x
                        .iter()
                        .zip(target.iter())
                        .map(|(xi, ti)| 2.0 * (xi - ti))
                        .collect::<Vec<_>>();
                    let f = x
                        .iter()
                        .zip(target.iter())
                        .map(|(xi, ti)| (xi - ti).powi(2))
                        .sum::<f64>();
                    Ok(Value::OutputList(vec![
                        Value::Num(f),
                        Value::Tensor(Tensor::new(grad, vec![3, 1]).unwrap()),
                    ]))
                })
            },
        )));
        let mut opts = StructValue::new();
        opts.insert("SpecifyObjectiveGradient", Value::Bool(true));
        opts.insert("TolX", Value::Num(1.0e-10));
        opts.insert("TolFun", Value::Num(1.0e-10));
        let _outputs = crate::output_count::push_output_count(Some(6));
        let result = block_on(fminunc_builtin(
            bound(104),
            tensor(vec![0.0, 0.0, 0.0]),
            vec![Value::Struct(opts)],
        ))
        .expect("fminunc");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 6);
        assert_close_vec(&vector_from_value(&outputs[0]), &[1.0, 2.0, 3.0], 1.0e-7);
        assert!(matches!(outputs[1], Value::Num(f) if f.abs() < 1.0e-14));
        assert!(matches!(outputs[2], Value::Num(flag) if flag > 0.0));
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
                assert!(
                    matches!(output.fields.get("firstorderopt"), Some(Value::Num(v)) if *v < 1.0e-8)
                );
                assert!(
                    matches!(output.fields.get("algorithm"), Some(Value::String(text)) if text.contains("bfgs"))
                );
            }
            other => panic!("unexpected output struct {other:?}"),
        }
        assert_close_vec(&vector_from_value(&outputs[4]), &[0.0, 0.0, 0.0], 1.0e-7);
        match &outputs[5] {
            Value::Tensor(hessian) => {
                assert_eq!(hessian.shape, vec![3, 3]);
                assert!(hessian.data.iter().all(|value| value.is_finite()));
                assert!(hessian.data[0] > 0.0);
                assert!(hessian.data[4] > 0.0);
                assert!(hessian.data[8] > 0.0);
            }
            other => panic!("unexpected hessian {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rejects_invalid_gradient_length() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, requested_outputs| {
                assert_eq!(requested_outputs, 2);
                Box::pin(async {
                    Ok(Value::OutputList(vec![
                        Value::Num(1.0),
                        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                    ]))
                })
            },
        )));
        let mut opts = StructValue::new();
        opts.insert("SpecifyObjectiveGradient", Value::Bool(true));
        let err = block_on(fminunc_builtin(
            bound(105),
            tensor(vec![0.0, 0.0, 0.0]),
            vec![Value::Struct(opts)],
        ))
        .unwrap_err();
        assert!(err.message().contains("gradient length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_scalar_input_returns_scalar() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num((x - 4.0).powi(2))) })
            },
        )));
        let result =
            block_on(fminunc_builtin(bound(106), Value::Num(0.0), Vec::new())).expect("fminunc");
        assert!(matches!(result, Value::Num(x) if (x - 4.0).abs() < 1.0e-4));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rejects_non_struct_options() {
        let err = block_on(fminunc_builtin(
            bound(107),
            Value::Num(0.0),
            vec![Value::Int(IntValue::I32(1))],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:fminunc:InvalidArgument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rejects_budget_too_small_for_finite_difference_gradient() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| {
                Box::pin(async { panic!("budget check should happen before invoking objective") })
            },
        )));
        let mut opts = StructValue::new();
        opts.insert("MaxFunEvals", Value::Num(3.0));
        let err = block_on(fminunc_builtin(
            bound(108),
            tensor(vec![0.0, 0.0, 0.0]),
            vec![Value::Struct(opts)],
        ))
        .unwrap_err();
        assert!(err.message().contains("MaxFunEvals"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rejects_problem_too_large_for_dense_bfgs() {
        let variables = MAX_DENSE_BFGS_VARIABLES + 1;
        let x0 = Value::Tensor(Tensor::new(vec![0.0; variables], vec![variables, 1]).unwrap());
        let err = block_on(fminunc_builtin(bound(109), x0, Vec::new())).unwrap_err();
        assert!(err.message().contains("dense BFGS"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fminunc_rejects_huge_direct_iteration_limits() {
        let mut opts = StructValue::new();
        opts.insert("MaxFunEvals", Value::Num((MAX_FUN_EVAL_LIMIT as f64) + 1.0));
        let err = block_on(fminunc_builtin(
            bound(110),
            Value::Num(0.0),
            vec![Value::Struct(opts)],
        ))
        .unwrap_err();
        assert!(err.message().contains("MaxFunEvals"));
    }
}
