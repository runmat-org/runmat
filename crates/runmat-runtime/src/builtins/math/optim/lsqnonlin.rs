//! MATLAB-compatible `lsqnonlin` builtin for nonlinear least-squares problems.

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
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{
    call_function, initial_guess, lookup_option, option_f64, option_string, option_usize,
};
use crate::builtins::math::optim::least_squares::{
    solve_least_squares, LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions,
    LeastSquaresResult, ResidualFuture,
};
use crate::builtins::math::optim::type_resolvers::nonlinear_solve_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lsqnonlin";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS_FACTOR: usize = 100;
const DEFAULT_ALGORITHM: &str = "trust-region-reflective";

const fn output_x() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solution with the same shape as x0.",
    }
}

const fn output_resnorm() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "resnorm",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Squared 2-norm of the final residual.",
    }
}

const fn output_residual() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "residual",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Final residual vector or array fun(x).",
    }
}

const fn output_exitflag() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solver exit condition.",
    }
}

const fn output_output() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "output",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Iteration and convergence metadata struct.",
    }
}

const fn output_lambda() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "lambda",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Approximate bound multiplier struct with lower and upper fields.",
    }
}

const fn output_jacobian() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "jacobian",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite-difference Jacobian of fun at the solution.",
    }
}

const OUTPUT_X: [BuiltinParamDescriptor; 1] = [output_x()];
const OUTPUT_X_RESNORM: [BuiltinParamDescriptor; 2] = [output_x(), output_resnorm()];
const OUTPUT_X_RESNORM_RESIDUAL: [BuiltinParamDescriptor; 3] =
    [output_x(), output_resnorm(), output_residual()];
const OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG: [BuiltinParamDescriptor; 4] = [
    output_x(),
    output_resnorm(),
    output_residual(),
    output_exitflag(),
];
const OUTPUT_CORE: [BuiltinParamDescriptor; 5] = [
    output_x(),
    output_resnorm(),
    output_residual(),
    output_exitflag(),
    output_output(),
];
const OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG_OUTPUT_LAMBDA: [BuiltinParamDescriptor; 6] = [
    output_x(),
    output_resnorm(),
    output_residual(),
    output_exitflag(),
    output_output(),
    output_lambda(),
];
const OUTPUT_ALL: [BuiltinParamDescriptor; 7] = [
    output_x(),
    output_resnorm(),
    output_residual(),
    output_exitflag(),
    output_output(),
    output_lambda(),
    output_jacobian(),
];

const fn input_fun() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Residual callback evaluated as fun(x).",
    }
}

const fn input_x0() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial design-variable guess.",
    }
}

const fn input_lb() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "lb",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Lower bounds. Empty entries mean unbounded; shorter vectors apply to leading variables.",
    }
}

const fn input_ub() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "ub",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Upper bounds. Empty entries mean unbounded; shorter vectors apply to leading variables.",
    }
}

const fn input_options() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimset or optimoptions.",
    }
}

const fn input_problem() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "problem",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Problem struct with solver, objective, x0, bounds, and options fields.",
    }
}

const INPUTS_CORE: [BuiltinParamDescriptor; 2] = [input_fun(), input_x0()];
const INPUTS_BOUNDS: [BuiltinParamDescriptor; 4] =
    [input_fun(), input_x0(), input_lb(), input_ub()];
const INPUTS_BOUNDS_OPTIONS: [BuiltinParamDescriptor; 5] = [
    input_fun(),
    input_x0(),
    input_lb(),
    input_ub(),
    input_options(),
];
const INPUTS_PROBLEM: [BuiltinParamDescriptor; 1] = [input_problem()];

const SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0, lb, ub)",
        inputs: &INPUTS_BOUNDS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0, lb, ub, options)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(problem)",
        inputs: &INPUTS_PROBLEM,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_CORE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output, lambda] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG_OUTPUT_LAMBDA,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0, lb, ub, [], [], [], [], [], options)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqnonlin:InvalidArgument"),
    when: "Argument grammar, bounds, options, constraints, problem struct, or output arity are invalid.",
    message: "lsqnonlin: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.INVALID_INPUT",
    identifier: Some("RunMat:lsqnonlin:InvalidInput"),
    when: "Initial guess, residual callback, or solver semantics are invalid.",
    message: "lsqnonlin: invalid input",
};

const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:lsqnonlin:TooManyOutputs"),
    when: "`lsqnonlin` is called with more than seven requested outputs.",
    message: "lsqnonlin: too many output arguments",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT,
    ERROR_TOO_MANY_OUTPUTS,
];

pub const LSQNONLIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn lsq_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("lsqnonlin:") {
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

fn map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        lsq_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::lsqnonlin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lsqnonlin",
    op_kind: GpuOpKind::Custom("nonlinear-least-squares"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host finite-difference Levenberg-Marquardt solver. Callback computations may use GPU-aware builtins, but residuals are gathered for the iterative solve.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::lsqnonlin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lsqnonlin",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Nonlinear least-squares solving repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "lsqnonlin",
    category = "math/optim",
    summary = "Solve nonlinear least-squares residual problems.",
    keywords = "lsqnonlin,least squares,optimization,levenberg-marquardt,bounds,residual,jacobian",
    accel = "sink",
    type_resolver(nonlinear_solve_type),
    descriptor(crate::builtins::math::optim::lsqnonlin::LSQNONLIN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::lsqnonlin"
)]
async fn lsqnonlin_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    validate_requested_outputs()?;
    let parsed = ParsedProblem::parse(first, rest)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let ParsedProblem {
        function,
        x0,
        lb,
        ub,
        options,
    } = parsed;
    let guess = initial_guess(NAME, x0)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
    let n = guess.values.len();
    let options = LsqOptions::from_struct(options.as_ref(), n)
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let bounds = ParsedProblem::bounds_from_values(lb, ub, n)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let mut evaluator = NonlinEvaluator {
        function,
        x_shape: guess.shape.clone(),
        x_scalar: guess.scalar,
        residual_shape: None,
    };
    let result = solve_least_squares(NAME, &mut evaluator, guess.values, &bounds, &options.solver)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
    let residual_shape = evaluator
        .residual_shape
        .as_deref()
        .unwrap_or(&[result.residual_len])
        .to_vec();
    finalize(
        result,
        &guess.shape,
        guess.scalar,
        &residual_shape,
        &bounds,
        &options.algorithm,
    )
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 7) {
        return Err(lsq_error_with_detail(
            &ERROR_TOO_MANY_OUTPUTS,
            "lsqnonlin: too many output arguments; maximum is 7",
        ));
    }
    Ok(())
}

struct ParsedProblem {
    function: Value,
    x0: Value,
    lb: Option<Value>,
    ub: Option<Value>,
    options: Option<StructValue>,
}

impl ParsedProblem {
    async fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        if rest.is_empty() {
            if let Value::Struct(problem) = first {
                return Self::from_problem_struct(problem);
            }
        }
        Self::from_function_args(first, rest).await
    }

    async fn from_function_args(function: Value, mut rest: Vec<Value>) -> BuiltinResult<Self> {
        if rest.is_empty() {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "lsqnonlin: expected initial guess x0",
            ));
        }
        let x0 = rest.remove(0);
        match rest.len() {
            0 => Ok(Self {
                function,
                x0,
                lb: None,
                ub: None,
                options: None,
            }),
            2 | 3 => {
                let mut values = rest.into_iter();
                let lb = values.next();
                let ub = values.next();
                let options = parse_options(values.next())?;
                Ok(Self {
                    function,
                    x0,
                    lb,
                    ub,
                    options,
                })
            }
            6..=8 => {
                let mut values = rest.into_iter();
                let lb = values.next();
                let ub = values.next();
                let a = values.next();
                let b = values.next();
                let aeq = values.next();
                let beq = values.next();
                ensure_empty_constraint("A", a.as_ref()).await?;
                ensure_empty_constraint("b", b.as_ref()).await?;
                ensure_empty_constraint("Aeq", aeq.as_ref()).await?;
                ensure_empty_constraint("beq", beq.as_ref()).await?;
                if let Some(nonlcon) = values.next() {
                    ensure_empty_constraint("nonlcon", Some(&nonlcon)).await?;
                }
                let options = parse_options(values.next())?;
                Ok(Self {
                    function,
                    x0,
                    lb,
                    ub,
                    options,
                })
            }
            _ => Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "lsqnonlin: expected fun,x0 or fun,x0,lb,ub[,options]",
            )),
        }
    }

    fn from_problem_struct(problem: StructValue) -> BuiltinResult<Self> {
        if let Some(solver) = lookup_option(&problem, "solver") {
            let solver = string_scalar("problem.solver", solver)?;
            if !solver.eq_ignore_ascii_case(NAME) {
                return Err(lsq_error_with_detail(
                    &ERROR_INVALID_ARGUMENT,
                    "problem.solver must be 'lsqnonlin'",
                ));
            }
        }
        if has_nonempty_problem_field(&problem, "A")
            || has_nonempty_problem_field(&problem, "b")
            || has_nonempty_problem_field(&problem, "Aineq")
            || has_nonempty_problem_field(&problem, "bineq")
            || has_nonempty_problem_field(&problem, "Aeq")
            || has_nonempty_problem_field(&problem, "beq")
            || has_nonempty_problem_field(&problem, "nonlcon")
        {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "linear and nonlinear constraints require the interior-point infrastructure and are not supported yet",
            ));
        }
        let function = lookup_option(&problem, "objective")
            .or_else(|| lookup_option(&problem, "fun"))
            .cloned()
            .ok_or_else(|| {
                lsq_error_with_detail(&ERROR_INVALID_ARGUMENT, "problem.objective is required")
            })?;
        let x0 = lookup_option(&problem, "x0").cloned().ok_or_else(|| {
            lsq_error_with_detail(&ERROR_INVALID_ARGUMENT, "problem.x0 is required")
        })?;
        let lb = lookup_option(&problem, "lb").cloned();
        let ub = lookup_option(&problem, "ub").cloned();
        let options = match lookup_option(&problem, "options") {
            None => None,
            Some(Value::Struct(options)) => Some(options.clone()),
            Some(other) if is_empty_value(other) => None,
            Some(other) => {
                return Err(lsq_error_with_detail(
                    &ERROR_INVALID_ARGUMENT,
                    format!("problem.options must be a struct, got {other:?}"),
                ))
            }
        };
        Ok(Self {
            function,
            x0,
            lb,
            ub,
            options,
        })
    }

    async fn bounds_from_values(
        lb: Option<Value>,
        ub: Option<Value>,
        n: usize,
    ) -> BuiltinResult<LeastSquaresBounds> {
        let mut bounds = LeastSquaresBounds::unbounded(n);
        if let Some(lb) = lb {
            bounds.lower = bound_vector("lower bounds", lb, n, f64::NEG_INFINITY).await?;
        }
        if let Some(ub) = ub {
            bounds.upper = bound_vector("upper bounds", ub, n, f64::INFINITY).await?;
        }
        bounds.validate(NAME, n)?;
        Ok(bounds)
    }
}

fn parse_options(value: Option<Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options)),
        Some(other) if is_empty_value(&other) => Ok(None),
        Some(other) => Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

async fn ensure_empty_constraint(label: &str, value: Option<&Value>) -> BuiltinResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
    if is_empty_value(&gathered) {
        Ok(())
    } else {
        Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("{label} constraints require the interior-point infrastructure and are not supported yet"),
        ))
    }
}

fn has_nonempty_problem_field(problem: &StructValue, name: &str) -> bool {
    lookup_option(problem, name).is_some_and(|value| !is_empty_value(value))
}

fn string_scalar(label: &str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::StringArray(values) if values.data.len() == 1 => Ok(values.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("{label} must be a character vector or string scalar, got {other:?}"),
        )),
    }
}

struct LsqOptions {
    solver: LeastSquaresOptions,
    algorithm: String,
}

impl LsqOptions {
    fn from_struct(options: Option<&StructValue>, _variable_len: usize) -> BuiltinResult<Self> {
        let display = option_string(options, "Display", "off")?;
        if !matches!(display.as_str(), "off" | "none" | "final" | "iter") {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let algorithm = option_string(options, "Algorithm", DEFAULT_ALGORITHM)?;
        let algorithm = algorithm.to_ascii_lowercase();
        if !matches!(
            algorithm.as_str(),
            "levenberg-marquardt" | "trust-region-reflective"
        ) {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "option Algorithm must be 'levenberg-marquardt' or 'trust-region-reflective'",
            ));
        }
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        let tol_fun = option_f64(NAME, options, "TolFun", DEFAULT_TOL_FUN)?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = option_usize(NAME, options, "MaxIter", DEFAULT_MAX_ITER)?.max(1);
        let max_fun_evals = option_usize(
            NAME,
            options,
            "MaxFunEvals",
            DEFAULT_MAX_FUN_EVALS_FACTOR * max_iter,
        )?
        .max(1);
        Ok(Self {
            solver: LeastSquaresOptions {
                tol_x,
                tol_fun,
                max_iter,
                max_fun_evals,
                final_jacobian: true,
            },
            algorithm,
        })
    }
}

struct NonlinEvaluator {
    function: Value,
    x_shape: Vec<usize>,
    x_scalar: bool,
    residual_shape: Option<Vec<usize>>,
}

impl LeastSquaresEvaluator for NonlinEvaluator {
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            let arg = x_value(x, &self.x_shape, self.x_scalar)?;
            let value = call_function(&self.function, vec![arg]).await?;
            let residual = real_array("function output", value).await?;
            if let Some(expected) = &self.residual_shape {
                if expected != &residual.shape {
                    return Err(lsq_error_with_detail(
                        &ERROR_INVALID_INPUT,
                        format!(
                            "function output shape {:?} changed from initial shape {:?}",
                            residual.shape, expected
                        ),
                    ));
                }
            } else {
                self.residual_shape = Some(residual.shape.clone());
            }
            Ok(residual.values)
        })
    }
}

#[derive(Clone)]
struct RealArray {
    values: Vec<f64>,
    shape: Vec<usize>,
}

async fn real_array(label: &str, value: Value) -> BuiltinResult<RealArray> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) => finite_array(label, vec![n], vec![1, 1]),
        Value::Int(i) => finite_array(label, vec![i.to_f64()], vec![1, 1]),
        Value::Bool(flag) => finite_array(label, vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            finite_array(label, tensor::tensor_into_values_f64(tensor), shape)
        }
        Value::LogicalArray(LogicalArray { data, shape }) => finite_array(
            label,
            data.into_iter()
                .map(|flag| if flag == 0 { 0.0 } else { 1.0 })
                .collect(),
            shape,
        ),
        other => Err(lsq_error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must be real numeric, got {other:?}"),
        )),
    }
}

fn finite_array(label: &str, values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<RealArray> {
    if values.is_empty() {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must not be empty"),
        ));
    }
    if let Some(value) = values.iter().find(|value| !value.is_finite()) {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("{label} must be finite, got {value}"),
        ));
    }
    Ok(RealArray { values, shape })
}

fn x_value(x: &[f64], shape: &[usize], scalar: bool) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(x[0]))
    } else {
        Tensor::new(x.to_vec(), shape.to_vec())
            .map(Value::Tensor)
            .map_err(|err| lsq_error_with_detail(&ERROR_INVALID_INPUT, err))
    }
}

async fn bound_vector(
    label: &str,
    value: Value,
    n: usize,
    default: f64,
) -> BuiltinResult<Vec<f64>> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    if is_empty_value(&value) {
        return Ok(vec![default; n]);
    }
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(flag) => vec![if flag { 1.0 } else { 0.0 }],
        Value::Tensor(Tensor { data, .. }) => data,
        Value::LogicalArray(LogicalArray { data, .. }) => data
            .into_iter()
            .map(|flag| if flag == 0 { 0.0 } else { 1.0 })
            .collect(),
        other => {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                format!("{label} must be real numeric or empty, got {other:?}"),
            ))
        }
    };
    if values.len() > n {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("{label} must be empty or contain no more elements than x0"),
        ));
    }
    let mut out = vec![default; n];
    for (dst, src) in out.iter_mut().zip(values) {
        *dst = src;
    }
    if out.iter().any(|value| value.is_nan()) {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("{label} must not contain NaN"),
        ));
    }
    Ok(out)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(Tensor { data, .. }) => data.is_empty(),
        Value::LogicalArray(LogicalArray { data, .. }) => data.is_empty(),
        _ => false,
    }
}

fn finalize(
    result: LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    residual_shape: &[usize],
    bounds: &LeastSquaresBounds,
    algorithm: &str,
) -> BuiltinResult<Value> {
    let x = x_value(&result.x, x_shape, x_scalar)?;
    let resnorm = Value::Num(result.resnorm);
    let residual = tensor_value(result.residual.clone(), residual_shape.to_vec())?;
    let exitflag = Value::Num(result.exitflag as f64);
    let output = Value::Struct(output_struct(&result, algorithm));
    let lambda = Value::Struct(lambda_struct(&result, x_shape, x_scalar, bounds)?);
    let jacobian = jacobian_value(&result)?;

    let outputs = match crate::output_count::current_output_count() {
        None => return Ok(x),
        Some(0) => return Ok(Value::OutputList(Vec::new())),
        Some(1) => vec![x],
        Some(2) => vec![x, resnorm],
        Some(3) => vec![x, resnorm, residual],
        Some(4) => vec![x, resnorm, residual, exitflag],
        Some(5) => vec![x, resnorm, residual, exitflag, output],
        Some(6) => vec![x, resnorm, residual, exitflag, output, lambda],
        Some(7) => vec![x, resnorm, residual, exitflag, output, lambda, jacobian],
        Some(_) => {
            return Err(lsq_error_with_detail(
                &ERROR_TOO_MANY_OUTPUTS,
                "lsqnonlin: too many output arguments; maximum is 7",
            ))
        }
    };
    Ok(crate::output_count::output_list_with_padding(
        outputs.len(),
        outputs,
    ))
}

fn tensor_value(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|err| lsq_error_with_detail(&ERROR_INVALID_INPUT, err))
}

fn output_struct(result: &LeastSquaresResult, algorithm: &str) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("firstorderopt", Value::Num(result.first_order_optimality));
    fields.insert("stepsize", Value::Num(result.step_size));
    fields.insert("cgiterations", Value::Num(0.0));
    fields.insert(
        "lssteplength",
        Value::Num(if result.step_size == 0.0 { 0.0 } else { 1.0 }),
    );
    fields.insert("algorithm", Value::from(algorithm));
    fields.insert("message", Value::from(result.message.clone()));
    fields
}

fn lambda_struct(
    result: &LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    bounds: &LeastSquaresBounds,
) -> BuiltinResult<StructValue> {
    let gradient = gradient(&result.jacobian, &result.residual, result.variable_len);
    let mut lower = vec![0.0; result.variable_len];
    let mut upper = vec![0.0; result.variable_len];
    for i in 0..result.variable_len {
        if bounds.lower[i].is_finite()
            && (result.x[i] - bounds.lower[i]).abs() <= 1.0e-8 * (1.0 + result.x[i].abs())
        {
            lower[i] = gradient[i].max(0.0);
        }
        if bounds.upper[i].is_finite()
            && (result.x[i] - bounds.upper[i]).abs() <= 1.0e-8 * (1.0 + result.x[i].abs())
        {
            upper[i] = (-gradient[i]).max(0.0);
        }
    }
    let mut fields = StructValue::new();
    fields.insert("lower", multiplier_value(lower, x_shape, x_scalar)?);
    fields.insert("upper", multiplier_value(upper, x_shape, x_scalar)?);
    Ok(fields)
}

fn multiplier_value(values: Vec<f64>, x_shape: &[usize], x_scalar: bool) -> BuiltinResult<Value> {
    if x_scalar {
        Ok(Value::Num(values[0]))
    } else {
        tensor_value(values, x_shape.to_vec())
    }
}

fn jacobian_value(result: &LeastSquaresResult) -> BuiltinResult<Value> {
    let rows = result.residual_len;
    let cols = result.variable_len;
    let mut column_major = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            column_major.push(result.jacobian[row * cols + col]);
        }
    }
    tensor_value(column_major, vec![rows, cols])
}

fn gradient(jacobian: &[f64], residual: &[f64], n: usize) -> Vec<f64> {
    let m = residual.len();
    let mut out = vec![0.0; n];
    for col in 0..n {
        for row in 0..m {
            out[col] += jacobian[row * n + col] * residual[row];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;
    use std::sync::Arc;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn lsqnonlin_real_array_reads_typed_integer_storage_exactly() {
        let mut input =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).expect("integer");
        input.data.fill(f64::NAN);

        let parsed = block_on(real_array("x0", Value::Tensor(input))).expect("real array");

        assert_eq!(parsed.values, vec![1.0, 2.0]);
        assert_eq!(parsed.shape, vec![2, 1]);
    }

    #[test]
    fn lsqnonlin_vector_residual_recovers_parameters() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected x tensor, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        vec![x[0] - 2.0, x[1] - 3.0, x[0] + x[1] - 5.0],
                        vec![3, 1],
                    ))
                })
            },
        )));
        let result = block_on(lsqnonlin_builtin(
            Value::BoundFunctionHandle {
                name: "residual".to_string(),
                function: 1,
            },
            vec![tensor(vec![0.0, 0.0], vec![2, 1])],
        ))
        .expect("lsqnonlin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 2.0).abs() < 1.0e-5);
                assert!((t.data[1] - 3.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_respects_bounds_and_empty_constraint_placeholders() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar x, got {other:?}"),
                };
                Box::pin(async move { Ok(tensor(vec![x - 2.0, x - 2.0], vec![1, 2])) })
            },
        )));
        let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
        let result = block_on(lsqnonlin_builtin(
            Value::BoundFunctionHandle {
                name: "bounded".to_string(),
                function: 2,
            },
            vec![
                Value::Num(0.0),
                Value::Num(-1.0),
                Value::Num(1.0),
                empty.clone(),
                empty.clone(),
                empty.clone(),
                empty.clone(),
                empty,
            ],
        ))
        .expect("lsqnonlin");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_accepts_empty_options_placeholder() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar x, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x - 3.0)) })
            },
        )));
        let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
        let result = block_on(lsqnonlin_builtin(
            Value::BoundFunctionHandle {
                name: "empty_options".to_string(),
                function: 22,
            },
            vec![Value::Num(0.0), empty.clone(), empty.clone(), empty],
        ))
        .expect("lsqnonlin accepts [] options");
        match result {
            Value::Num(value) => assert!((value - 3.0).abs() < 1.0e-5),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_partial_bounds_apply_to_leading_variables() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected x tensor, got {other:?}"),
                };
                Box::pin(async move { Ok(tensor(vec![x[0] - 2.0, x[1] - 5.0], vec![2, 1])) })
            },
        )));
        let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
        let result = block_on(lsqnonlin_builtin(
            Value::BoundFunctionHandle {
                name: "partial_bounds".to_string(),
                function: 23,
            },
            vec![tensor(vec![0.0, 0.0], vec![2, 1]), empty, Value::Num(1.0)],
        ))
        .expect("lsqnonlin partial bounds");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 1.0).abs() < 1.0e-5);
                assert!((t.data[1] - 5.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_multi_output_includes_diagnostics_lambda_and_jacobian() {
        let _guard = crate::output_count::push_output_count(Some(7));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected x tensor, got {other:?}"),
                };
                Box::pin(async move { Ok(tensor(vec![x[0] - 1.0, 2.0 * x[1] - 4.0], vec![1, 2])) })
            },
        )));
        let mut options = StructValue::new();
        options.insert("Algorithm", Value::from("levenberg-marquardt"));
        let result = block_on(lsqnonlin_builtin(
            Value::BoundFunctionHandle {
                name: "multi".to_string(),
                function: 3,
            },
            vec![
                tensor(vec![0.0, 0.0], vec![1, 2]),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Struct(options),
            ],
        ))
        .expect("lsqnonlin");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 7);
                assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
                assert!(matches!(&outputs[2], Value::Tensor(t) if t.shape == vec![1, 2]));
                assert!(matches!(&outputs[3], Value::Num(flag) if *flag > 0.0));
                assert!(
                    matches!(&outputs[4], Value::Struct(output) if output.fields.contains_key("funcCount") && output.fields.contains_key("stepsize") && output.fields.contains_key("cgiterations") && output.fields.contains_key("lssteplength"))
                );
                assert!(
                    matches!(&outputs[5], Value::Struct(lambda) if lambda.fields.contains_key("lower") && lambda.fields.contains_key("upper"))
                );
                match &outputs[6] {
                    Value::Tensor(j) => {
                        assert_eq!(j.shape, vec![2, 2]);
                        let expected = [1.0, 0.0, 0.0, 2.0];
                        for (actual, expected) in j.data.iter().zip(expected) {
                            assert!((actual - expected).abs() < 1.0e-6);
                        }
                    }
                    other => panic!("expected jacobian tensor, got {other:?}"),
                }
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_accepts_problem_struct() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected x tensor, got {other:?}"),
                };
                Box::pin(async move { Ok(tensor(vec![x[0] - 4.0, x[1] - 5.0], vec![2, 1])) })
            },
        )));
        let mut problem = StructValue::new();
        problem.insert("solver", Value::from("lsqnonlin"));
        problem.insert(
            "objective",
            Value::BoundFunctionHandle {
                name: "problem".to_string(),
                function: 4,
            },
        );
        problem.insert("x0", tensor(vec![0.0, 0.0], vec![2, 1]));
        problem.insert("lb", tensor(vec![1.0, 2.0], vec![2, 1]));
        problem.insert("ub", tensor(vec![10.0, 10.0], vec![2, 1]));
        let result = block_on(lsqnonlin_builtin(Value::Struct(problem), Vec::new()))
            .expect("lsqnonlin problem");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                assert!(
                    matches!(&outputs[0], Value::Tensor(t) if (t.data[0] - 4.0).abs() < 1.0e-5 && (t.data[1] - 5.0).abs() < 1.0e-5)
                );
                assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqnonlin_problem_struct_rejects_aineq_and_bineq() {
        let mut problem = StructValue::new();
        problem.insert("solver", Value::from("lsqnonlin"));
        problem.insert("objective", Value::FunctionHandle("sin".into()));
        problem.insert("x0", Value::Num(0.0));
        problem.insert("Aineq", Value::Num(1.0));
        problem.insert("bineq", Value::Num(0.0));
        let err = block_on(lsqnonlin_builtin(Value::Struct(problem), Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:InvalidArgument"));
        assert!(err.message().contains("constraints"));
    }

    #[test]
    fn lsqnonlin_default_max_fun_evals_matches_optimoptions() {
        let options = LsqOptions::from_struct(None, 7).expect("default options");
        assert_eq!(
            options.solver.max_fun_evals,
            DEFAULT_MAX_FUN_EVALS_FACTOR * DEFAULT_MAX_ITER
        );
    }

    #[test]
    fn lsqnonlin_rejects_real_constraints_until_interior_point_exists() {
        let err = block_on(lsqnonlin_builtin(
            Value::FunctionHandle("sin".into()),
            vec![
                Value::Num(0.0),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Num(1.0),
                Value::Num(0.0),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:InvalidArgument"));
    }

    #[test]
    fn lsqnonlin_rejects_more_than_seven_outputs() {
        let _guard = crate::output_count::push_output_count(Some(8));
        let err = block_on(lsqnonlin_builtin(
            Value::FunctionHandle("sin".into()),
            vec![Value::Num(0.0)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:TooManyOutputs"));
    }
}
