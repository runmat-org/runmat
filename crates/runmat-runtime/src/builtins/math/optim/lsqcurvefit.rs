//! MATLAB-compatible `lsqcurvefit` builtin for nonlinear curve fitting.

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
    call_function, initial_guess, option_f64, option_string, option_usize,
};
use crate::builtins::math::optim::least_squares::{
    solve_least_squares, LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions,
    LeastSquaresResult, ResidualFuture,
};
use crate::builtins::math::optim::type_resolvers::nonlinear_solve_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lsqcurvefit";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS_FACTOR: usize = 100;
const ALGORITHM: &str = "levenberg-marquardt";

const fn output_x() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated fit parameters with the same shape as x0.",
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
        description: "Final model residual fun(x,xdata)-ydata.",
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
        description: "Model callback evaluated as fun(x,xdata).",
    }
}

const fn input_x0() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial parameter guess.",
    }
}

const fn input_xdata() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "xdata",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Independent data passed to the model callback.",
    }
}

const fn input_ydata() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "ydata",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Observed response data.",
    }
}

const fn input_lb() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "lb",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Lower parameter bounds. Empty means unbounded.",
    }
}

const fn input_ub() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "ub",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Upper parameter bounds. Empty means unbounded.",
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

const INPUTS_CORE: [BuiltinParamDescriptor; 4] =
    [input_fun(), input_x0(), input_xdata(), input_ydata()];

const INPUTS_BOUNDS: [BuiltinParamDescriptor; 6] = [
    input_fun(),
    input_x0(),
    input_xdata(),
    input_ydata(),
    input_lb(),
    input_ub(),
];

const INPUTS_BOUNDS_OPTIONS: [BuiltinParamDescriptor; 7] = [
    input_fun(),
    input_x0(),
    input_xdata(),
    input_ydata(),
    input_lb(),
    input_ub(),
    input_options(),
];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "x = lsqcurvefit(fun, x0, xdata, ydata)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub)",
        inputs: &INPUTS_BOUNDS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_CORE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output, lambda] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_X_RESNORM_RESIDUAL_EXITFLAG_OUTPUT_LAMBDA,
    },
    BuiltinSignatureDescriptor {
        label: "[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(___)",
        inputs: &INPUTS_BOUNDS_OPTIONS,
        outputs: &OUTPUT_ALL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqcurvefit:InvalidArgument"),
    when: "Argument grammar, bounds, options, or output arity are invalid.",
    message: "lsqcurvefit: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.INVALID_INPUT",
    identifier: Some("RunMat:lsqcurvefit:InvalidInput"),
    when: "Initial guess, model callback, data shape, or solver semantics are invalid.",
    message: "lsqcurvefit: invalid input",
};

const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:lsqcurvefit:TooManyOutputs"),
    when: "`lsqcurvefit` is called with more than seven requested outputs.",
    message: "lsqcurvefit: too many output arguments",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT,
    ERROR_TOO_MANY_OUTPUTS,
];

pub const LSQCURVEFIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
    let message = if detail.starts_with("lsqcurvefit:") {
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::lsqcurvefit")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lsqcurvefit",
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::lsqcurvefit")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lsqcurvefit",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Nonlinear curve fitting repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "lsqcurvefit",
    category = "math/optim",
    summary = "Fit nonlinear parametric curves in the least-squares sense.",
    keywords = "lsqcurvefit,least squares,curve fitting,optimization,levenberg-marquardt,bounds",
    accel = "sink",
    type_resolver(nonlinear_solve_type),
    descriptor(crate::builtins::math::optim::lsqcurvefit::LSQCURVEFIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::lsqcurvefit"
)]
async fn lsqcurvefit_builtin(
    function: Value,
    x0: Value,
    xdata: Value,
    ydata: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_requested_outputs()?;
    let parsed = ParsedArgs::parse(rest)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let guess = initial_guess(NAME, x0)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
    let n = guess.values.len();
    let options = LsqOptions::from_struct(parsed.options.as_ref())
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let bounds = parsed
        .bounds(n)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let ydata = real_array("ydata", ydata)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
    if ydata.values.is_empty() {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_INPUT,
            "ydata must not be empty",
        ));
    }
    let mut evaluator = CurveFitEvaluator {
        function,
        x_shape: guess.shape.clone(),
        x_scalar: guess.scalar,
        xdata,
        ydata,
    };
    let result = solve_least_squares(NAME, &mut evaluator, guess.values, &bounds, &options.solver)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
    finalize(
        result,
        &guess.shape,
        guess.scalar,
        &evaluator.ydata.shape,
        &bounds,
        &options.algorithm,
    )
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 7) {
        return Err(lsq_error_with_detail(
            &ERROR_TOO_MANY_OUTPUTS,
            "lsqcurvefit: too many output arguments; maximum is 7",
        ));
    }
    Ok(())
}

struct ParsedArgs {
    lb: Option<Value>,
    ub: Option<Value>,
    options: Option<StructValue>,
}

impl ParsedArgs {
    async fn parse(rest: Vec<Value>) -> BuiltinResult<Self> {
        match rest.len() {
            0 => Ok(Self {
                lb: None,
                ub: None,
                options: None,
            }),
            1 => match rest.into_iter().next().unwrap() {
                Value::Struct(options) => Ok(Self {
                    lb: None,
                    ub: None,
                    options: Some(options),
                }),
                other => Err(lsq_error_with_detail(
                    &ERROR_INVALID_ARGUMENT,
                    format!("single optional argument must be options struct, got {other:?}"),
                )),
            },
            2 | 3 => {
                let mut values = rest.into_iter();
                let lb = values.next();
                let ub = values.next();
                let options = match values.next() {
                    None => None,
                    Some(Value::Struct(options)) => Some(options),
                    Some(other) => {
                        return Err(lsq_error_with_detail(
                            &ERROR_INVALID_ARGUMENT,
                            format!("options must be a struct, got {other:?}"),
                        ))
                    }
                };
                Ok(Self { lb, ub, options })
            }
            _ => Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "linear and nonlinear constraint forms are not supported yet",
            )),
        }
    }

    async fn bounds(self, n: usize) -> BuiltinResult<LeastSquaresBounds> {
        let mut bounds = LeastSquaresBounds::unbounded(n);
        if let Some(lb) = self.lb {
            bounds.lower = bound_vector("lower bounds", lb, n, f64::NEG_INFINITY).await?;
        }
        if let Some(ub) = self.ub {
            bounds.upper = bound_vector("upper bounds", ub, n, f64::INFINITY).await?;
        }
        bounds.validate(NAME, n)?;
        Ok(bounds)
    }
}

struct LsqOptions {
    solver: LeastSquaresOptions,
    algorithm: String,
}

impl LsqOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        let display = option_string(options, "Display", "off")?;
        if !matches!(display.as_str(), "off" | "none" | "final" | "iter") {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let algorithm = option_string(options, "Algorithm", ALGORITHM)?;
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

struct CurveFitEvaluator {
    function: Value,
    x_shape: Vec<usize>,
    x_scalar: bool,
    xdata: Value,
    ydata: RealArray,
}

impl LeastSquaresEvaluator for CurveFitEvaluator {
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            let arg = x_value(x, &self.x_shape, self.x_scalar)?;
            let value = call_function(&self.function, vec![arg, self.xdata.clone()]).await?;
            let model = real_array("model output", value).await?;
            if model.shape != self.ydata.shape {
                return Err(lsq_error_with_detail(
                    &ERROR_INVALID_INPUT,
                    format!(
                        "model output shape {:?} must match ydata shape {:?}",
                        model.shape, self.ydata.shape
                    ),
                ));
            }
            Ok(model
                .values
                .iter()
                .zip(self.ydata.values.iter())
                .map(|(model, observed)| model - observed)
                .collect())
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
        Value::Tensor(tensor) => finite_array(label, tensor.data, tensor.shape),
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
    let out = if values.len() == 1 && n > 1 {
        vec![values[0]; n]
    } else if values.len() == n {
        values
    } else {
        return Err(lsq_error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            format!("{label} must be scalar, empty, or match x0 length"),
        ));
    };
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
                "lsqcurvefit: too many output arguments; maximum is 7",
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
    use std::sync::Arc;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn lsqcurvefit_linear_model_recovers_parameters() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let p = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected xdata, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        xdata.iter().map(|x| p[0] * x + p[1]).collect::<Vec<_>>(),
                        vec![1, xdata.len()],
                    ))
                })
            },
        )));
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "line".to_string(),
                function: 1,
            },
            tensor(vec![0.0, 0.0], vec![2, 1]),
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
            tensor(vec![1.0, 3.0, 5.0, 7.0], vec![1, 4]),
            Vec::new(),
        ))
        .expect("lsqcurvefit");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 2.0).abs() < 1.0e-5);
                assert!((t.data[1] - 1.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_exponential_model_recovers_parameters() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected xdata, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        xdata
                            .iter()
                            .map(|x| p[0] * (-p[1] * x).exp())
                            .collect::<Vec<_>>(),
                        vec![1, xdata.len()],
                    ))
                })
            },
        )));
        let xdata = (0..=10).map(|i| i as f64 * 0.2).collect::<Vec<_>>();
        let ydata = xdata
            .iter()
            .map(|x| 2.5 * (-0.7 * x).exp())
            .collect::<Vec<_>>();
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "exp_decay".to_string(),
                function: 2,
            },
            tensor(vec![1.0, 0.1], vec![2, 1]),
            tensor(xdata.clone(), vec![1, xdata.len()]),
            tensor(ydata, vec![1, xdata.len()]),
            Vec::new(),
        ))
        .expect("lsqcurvefit");
        match result {
            Value::Tensor(t) => {
                assert!((t.data[0] - 2.5).abs() < 1.0e-4);
                assert!((t.data[1] - 0.7).abs() < 1.0e-4);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_respects_bounds() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected xdata, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        xdata.iter().map(|x| p[0] * x).collect::<Vec<_>>(),
                        vec![1, xdata.len()],
                    ))
                })
            },
        )));
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "bounded_line".to_string(),
                function: 3,
            },
            tensor(vec![0.5], vec![1, 1]),
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            tensor(vec![2.0, 4.0, 6.0], vec![1, 3]),
            vec![Value::Tensor(Tensor::zeros(vec![0, 0])), Value::Num(1.0)],
        ))
        .expect("lsqcurvefit");
        match result {
            Value::Tensor(t) => assert!((t.data[0] - 1.0).abs() < 1.0e-8),
            Value::Num(n) => assert!((n - 1.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_multi_output_includes_diagnostics_lambda_and_jacobian() {
        let _guard = crate::output_count::push_output_count(Some(7));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected xdata, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        xdata.iter().map(|x| p[0] * x + p[1]).collect(),
                        vec![1, xdata.len()],
                    ))
                })
            },
        )));
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "line_multi".to_string(),
                function: 4,
            },
            tensor(vec![0.0, 0.0], vec![1, 2]),
            tensor(vec![0.0, 1.0, 2.0], vec![1, 3]),
            tensor(vec![1.0, 3.0, 5.0], vec![1, 3]),
            Vec::new(),
        ))
        .expect("lsqcurvefit");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 7);
                assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
                assert!(matches!(&outputs[2], Value::Tensor(t) if t.shape == vec![1, 3]));
                assert!(matches!(&outputs[3], Value::Num(flag) if *flag > 0.0));
                assert!(
                    matches!(&outputs[4], Value::Struct(output) if output.fields.contains_key("funcCount"))
                );
                assert!(
                    matches!(&outputs[5], Value::Struct(lambda) if lambda.fields.contains_key("lower") && lambda.fields.contains_key("upper"))
                );
                match &outputs[6] {
                    Value::Tensor(j) => {
                        assert_eq!(j.shape, vec![3, 2]);
                        let expected = [0.0, 1.0, 2.0, 1.0, 1.0, 1.0];
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
    fn lsqcurvefit_output_reports_selected_algorithm() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Num(p) => *p,
                    other => panic!("expected scalar param, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected xdata, got {other:?}"),
                };
                Box::pin(async move {
                    Ok(tensor(
                        xdata.iter().map(|x| p * x).collect::<Vec<_>>(),
                        vec![1, xdata.len()],
                    ))
                })
            },
        )));
        let mut options = StructValue::new();
        options.insert("Algorithm", Value::from("trust-region-reflective"));
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "line_algorithm".to_string(),
                function: 45,
            },
            Value::Num(0.0),
            tensor(vec![1.0, 2.0], vec![1, 2]),
            tensor(vec![2.0, 4.0], vec![1, 2]),
            vec![
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Struct(options),
            ],
        ))
        .expect("lsqcurvefit");
        match result {
            Value::OutputList(outputs) => match &outputs[4] {
                Value::Struct(output) => match output.fields.get("algorithm") {
                    Some(Value::String(algorithm)) => {
                        assert_eq!(algorithm, "trust-region-reflective")
                    }
                    other => panic!("expected algorithm string, got {other:?}"),
                },
                other => panic!("expected output struct, got {other:?}"),
            },
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_underdetermined_case_runs() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(p[0] + p[1])) })
            },
        )));
        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "underdetermined".to_string(),
                function: 5,
            },
            tensor(vec![0.0, 0.0], vec![2, 1]),
            Value::Num(0.0),
            Value::Num(3.0),
            Vec::new(),
        ))
        .expect("lsqcurvefit");
        match result {
            Value::Tensor(t) => assert!((t.data[0] + t.data[1] - 3.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_rejects_model_ydata_shape_mismatch() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| {
                Box::pin(async move { Ok(tensor(vec![1.0, 2.0], vec![1, 2])) })
            },
        )));
        let err = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "bad_shape".to_string(),
                function: 6,
            },
            Value::Num(0.0),
            Value::Num(0.0),
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:InvalidInput"));
    }

    #[test]
    fn lsqcurvefit_rejects_inconsistent_bounds() {
        let err = block_on(lsqcurvefit_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(0.0),
            vec![Value::Num(2.0), Value::Num(1.0)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:InvalidArgument"));
    }

    #[test]
    fn lsqcurvefit_rejects_more_than_seven_outputs() {
        let _guard = crate::output_count::push_output_count(Some(8));
        let err = block_on(lsqcurvefit_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:TooManyOutputs"));
    }
}
