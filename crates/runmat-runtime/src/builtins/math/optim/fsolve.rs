//! MATLAB-compatible `fsolve` builtin for nonlinear systems.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{
    call_function, option_f64, option_string, option_usize, value_to_real_vector, vector_to_value,
};
use crate::builtins::math::optim::least_squares::{
    solve_least_squares, LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions,
    LeastSquaresResult, ResidualFuture,
};
use crate::builtins::math::optim::type_resolvers::nonlinear_solve_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fsolve";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;

const FSOLVE_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Approximate solution vector/scalar.",
}];

macro_rules! fsolve_output {
    ($name:literal, $ty:expr, $description:literal) => {
        BuiltinParamDescriptor {
            name: $name,
            ty: $ty,
            arity: BuiltinParamArity::Required,
            default: None,
            description: $description,
        }
    };
}
const FSOLVE_OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [
    fsolve_output!("x", BuiltinParamType::NumericArray, "Approximate solution."),
    fsolve_output!(
        "fval",
        BuiltinParamType::NumericArray,
        "Function value at x."
    ),
];
const FSOLVE_OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] = [
    fsolve_output!("x", BuiltinParamType::NumericArray, "Approximate solution."),
    fsolve_output!(
        "fval",
        BuiltinParamType::NumericArray,
        "Function value at x."
    ),
    fsolve_output!(
        "exitflag",
        BuiltinParamType::NumericScalar,
        "Convergence status code."
    ),
];
const FSOLVE_OUTPUT_X_FVAL_EXITFLAG_OUTPUT: [BuiltinParamDescriptor; 4] = [
    fsolve_output!("x", BuiltinParamType::NumericArray, "Approximate solution."),
    fsolve_output!(
        "fval",
        BuiltinParamType::NumericArray,
        "Function value at x."
    ),
    fsolve_output!(
        "exitflag",
        BuiltinParamType::NumericScalar,
        "Convergence status code."
    ),
    fsolve_output!("output", BuiltinParamType::Any, "Solver diagnostics."),
];
const FSOLVE_OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
    fsolve_output!("x", BuiltinParamType::NumericArray, "Approximate solution."),
    fsolve_output!(
        "fval",
        BuiltinParamType::NumericArray,
        "Function value at x."
    ),
    fsolve_output!(
        "exitflag",
        BuiltinParamType::NumericScalar,
        "Convergence status code."
    ),
    fsolve_output!("output", BuiltinParamType::Any, "Solver diagnostics."),
    fsolve_output!("jacobian", BuiltinParamType::NumericArray, "Jacobian at x."),
];

const FSOLVE_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "System residual callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial guess scalar/vector.",
    },
];

const FSOLVE_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "System residual callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial guess scalar/vector.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimset.",
    },
];

const FSOLVE_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "x = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval] = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval] = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag] = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag] = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag,output] = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_X_FVAL_EXITFLAG_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag,output] = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_X_FVAL_EXITFLAG_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_ALL,
    },
];

const FSOLVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSOLVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:fsolve:InvalidArgument"),
    when: "Argument grammar/options configuration is invalid.",
    message: "fsolve: invalid argument",
};

const FSOLVE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSOLVE.INVALID_INPUT",
    identifier: Some("RunMat:fsolve:InvalidInput"),
    when: "Initial guess/callback/iteration semantics are invalid.",
    message: "fsolve: invalid input",
};

const FSOLVE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FSOLVE_ERROR_INVALID_ARGUMENT, FSOLVE_ERROR_INVALID_INPUT];

const FSOLVE_INTEGER_INITIAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x0",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer initial guesses are independently gated and every element must convert exactly to binary64.",
    }];
const FSOLVE_INTEGER_RESIDUAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fun(x) residual",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer residuals are gated before resident gather and every element must convert exactly to binary64.",
    }];
const FSOLVE_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "TolX",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolX is independently gated and must convert exactly to a positive binary64 scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "TolFun",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolFun is independently gated and must convert exactly to a positive binary64 scalar.",
    },
];
const FSOLVE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "MaxIter",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer iteration counts are independently gated and decoded exactly through platform bounds.",
    },
    BuiltinIntegerInputCapability {
        name: "MaxFunEvals",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer evaluation counts are independently gated and decoded exactly through platform bounds.",
    },
];
pub const FSOLVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = fsolve(fun, integer_x0, options)",
        inputs: &FSOLVE_INTEGER_INITIAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented x0 is double. Strict compatibility rejects typed integers; RunMat mode admits exact binary64 values and returns double x, fval, exitflag, diagnostics, and Jacobian outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fsolve callback returns an integer residual",
        inputs: &FSOLVE_INTEGER_RESIDUAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Strict compatibility rejects typed residuals before provider access. RunMat mode converts exact values to binary64; the one-to-five output contract remains function-specific and double-valued for numeric results.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fsolve(..., options.TolX=integer, options.TolFun=integer)",
        inputs: &FSOLVE_INTEGER_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer tolerance controls are RunMat-only and do not change output classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fsolve(..., options.MaxIter=integer, options.MaxFunEvals=integer)",
        inputs: &FSOLVE_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented controls are integer-valued numeric scalars, not documented typed-integer classes. RunMat's typed forms preserve exact counts without a binary64 round trip.",
    },
];

pub(crate) const FSOLVE_INPUT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsolve-nonfloating-initial-point",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsolve with a typed-integer or logical initial point is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsolveNumericInputExtension"),
    };
pub(crate) const FSOLVE_CALLBACK_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsolve-nonfloating-callback-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsolve with typed-integer or logical residual output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsolveCallbackExtension"),
    };
pub(crate) const FSOLVE_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsolve-typed-option-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsolve with typed-integer option controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsolveOptionExtension"),
};
pub(crate) const FSOLVE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsolve-resident-fallback",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fsolve with provider-resident numeric input or callback output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsolveResidentExtension"),
    };
pub const FSOLVE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FSOLVE_INPUT_NUMERIC_EXTENSION,
    FSOLVE_CALLBACK_NUMERIC_EXTENSION,
    FSOLVE_OPTION_EXTENSION,
    FSOLVE_RESIDENT_EXTENSION,
];

pub const FSOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FSOLVE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FSOLVE_ERRORS,
};

fn fsolve_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fsolve:") {
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

fn fsolve_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        fsolve_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fsolve")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fsolve",
    op_kind: GpuOpKind::Custom("nonlinear-solve"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host finite-difference Levenberg-Marquardt solver. Callback computations may use GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fsolve")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fsolve",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Nonlinear solving repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fsolve",
    category = "math/optim",
    summary = "Solve nonlinear equation systems.",
    keywords = "fsolve,nonlinear solve,root finding,levenberg-marquardt,jacobian",
    accel = "sink",
    type_resolver(nonlinear_solve_type),
    descriptor(crate::builtins::math::optim::fsolve::FSOLVE_DESCRIPTOR),
    extensions(crate::builtins::math::optim::fsolve::FSOLVE_EXTENSIONS),
    integer_capabilities(crate::builtins::math::optim::fsolve::FSOLVE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::optim::fsolve"
)]
async fn fsolve_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = parse_options(rest.first())
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let opts = FsolveOptions::from_struct(options.as_ref())
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let guess = crate::builtins::math::optim::common::initial_guess_with_extensions(
        NAME,
        x0,
        &FSOLVE_INPUT_NUMERIC_EXTENSION,
        &FSOLVE_RESIDENT_EXTENSION,
    )
    .await
    .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    let outcome = solve(&function, guess.values, &guess.shape, guess.scalar, &opts)
        .await
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    finalize(outcome, &guess.shape, guess.scalar)
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy)]
struct FsolveOptions {
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: usize,
}

impl FsolveOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        crate::builtins::math::optim::common::ensure_option_extensions(
            NAME,
            options,
            &FSOLVE_OPTION_EXTENSION,
            &FSOLVE_RESIDENT_EXTENSION,
        )?;
        let display = option_string(options, "Display", "off")?;
        if !matches!(display.as_str(), "off" | "none" | "final" | "iter") {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        let tol_fun = option_f64(NAME, options, "TolFun", DEFAULT_TOL_FUN)?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = option_usize(NAME, options, "MaxIter", DEFAULT_MAX_ITER)?.max(1);
        let max_fun_evals = option_usize(NAME, options, "MaxFunEvals", 100 * max_iter)?.max(1);
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
        })
    }
}

async fn solve(
    function: &Value,
    x: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FsolveOptions,
) -> BuiltinResult<FsolveOutcome> {
    let mut evaluator = FsolveEvaluator {
        function,
        shape: shape.to_vec(),
        scalar,
        residual_shape: None,
        residual_scalar: false,
    };
    let variable_len = x.len();
    let mut result = solve_least_squares(
        NAME,
        &mut evaluator,
        x,
        &LeastSquaresBounds::unbounded(variable_len),
        &LeastSquaresOptions {
            tol_x: options.tol_x,
            tol_fun: options.tol_fun,
            max_iter: options.max_iter,
            max_fun_evals: options.max_fun_evals,
            final_jacobian: true,
        },
    )
    .await?;
    if result.exitflag > 0
        && result
            .residual
            .iter()
            .fold(0.0_f64, |norm, value| norm.max(value.abs()))
            > options.tol_fun
    {
        result.exitflag = -2;
        result.message =
            "Equation not solved. The solver converged to a point with a nonzero residual."
                .to_string();
    }
    Ok(FsolveOutcome {
        result,
        residual_shape: evaluator.residual_shape.unwrap_or_else(|| vec![1, 1]),
        residual_scalar: evaluator.residual_scalar,
    })
}

struct FsolveOutcome {
    result: LeastSquaresResult,
    residual_shape: Vec<usize>,
    residual_scalar: bool,
}

struct FsolveEvaluator<'a> {
    function: &'a Value,
    shape: Vec<usize>,
    scalar: bool,
    residual_shape: Option<Vec<usize>>,
    residual_scalar: bool,
}

impl LeastSquaresEvaluator for FsolveEvaluator<'_> {
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            let arg = if self.scalar {
                Value::Num(x[0])
            } else {
                Value::Tensor(
                    runmat_builtins::Tensor::new(x.to_vec(), self.shape.clone())
                        .map_err(|e| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_INPUT, e))?,
                )
            };
            let value = call_function(self.function, vec![arg]).await?;
            let value = crate::builtins::math::optim::common::prepare_floating_value(
                NAME,
                value,
                &FSOLVE_CALLBACK_NUMERIC_EXTENSION,
                &FSOLVE_RESIDENT_EXTENSION,
                "function residual",
            )
            .await?;
            let (shape, scalar) = match &value {
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => (vec![1, 1], true),
                Value::Tensor(tensor) => (tensor.shape.clone(), false),
                Value::LogicalArray(array) => (array.shape.clone(), false),
                _ => (vec![1, 1], false),
            };
            self.residual_shape = Some(shape);
            self.residual_scalar = scalar;
            let residual = value_to_real_vector(NAME, value).await?;
            if residual.is_empty() {
                Err(fsolve_error_with_detail(
                    &FSOLVE_ERROR_INVALID_INPUT,
                    "function value must not be empty",
                ))
            } else {
                Ok(residual)
            }
        })
    }
}

fn finalize(outcome: FsolveOutcome, x_shape: &[usize], x_scalar: bool) -> BuiltinResult<Value> {
    let result = outcome.result;
    let x = vector_to_value(NAME, result.x.clone(), x_shape, x_scalar)?;
    let fval = vector_to_value(
        NAME,
        result.residual.clone(),
        &outcome.residual_shape,
        outcome.residual_scalar,
    )?;
    let exitflag = Value::Num(result.exitflag as f64);
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("algorithm", Value::from("levenberg-marquardt"));
    fields.insert("firstorderopt", Value::Num(result.first_order_optimality));
    fields.insert("stepsize", Value::Num(result.step_size));
    fields.insert("message", Value::from(result.message.clone()));
    let output = Value::Struct(fields);
    let mut jacobian_data = Vec::with_capacity(result.jacobian.len());
    for column in 0..result.variable_len {
        for row in 0..result.residual_len {
            jacobian_data.push(result.jacobian[row * result.variable_len + column]);
        }
    }
    let jacobian = Value::Tensor(
        runmat_builtins::Tensor::new(
            jacobian_data,
            vec![result.residual_len, result.variable_len],
        )
        .map_err(|error| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_INPUT, error))?,
    );
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
            vec![x, fval, exitflag, output, jacobian],
        )),
        Some(_) => Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            "too many output arguments; maximum is 5",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::{Arc, Mutex};

    #[test]
    fn fsolve_scalar_builtin_handle() {
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-5),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_five_output_form_returns_residual_status_diagnostics_and_jacobian() {
        let _outputs = crate::output_count::push_output_count(Some(5));
        let result = block_on(fsolve_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("expected five outputs");
        };
        assert_eq!(outputs.len(), 5);
        assert!(matches!(outputs[0], Value::Num(_)));
        assert!(matches!(outputs[1], Value::Num(_)));
        assert!(matches!(outputs[2], Value::Num(_)));
        assert!(matches!(outputs[3], Value::Struct(_)));
        assert!(matches!(&outputs[4], Value::Tensor(tensor) if tensor.shape == vec![1, 1]));
    }

    #[test]
    fn fsolve_returns_stationary_non_root_with_solver_status() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar numeric argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x * x + 1.0)) })
            },
        )));
        let _outputs = crate::output_count::push_output_count(Some(3));
        let result = block_on(fsolve_builtin(
            Value::BoundFunctionHandle {
                name: "no_real_root".to_string(),
                function: 44,
            },
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("expected outputs")
        };
        assert!(matches!(outputs[2], Value::Num(flag) if flag <= 0.0));
    }

    #[test]
    fn fsolve_vector_system_via_semantic_resolver() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(|_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    _ => panic!("expected tensor input"),
                };
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(
                            vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] * x[1] - 1.0],
                            vec![2, 1],
                        )
                        .unwrap(),
                    ))
                })
            }),
        ));
        let x0 = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert!(
                    (t.materialize_f64()[0] * t.materialize_f64()[0]
                        + t.materialize_f64()[1] * t.materialize_f64()[1]
                        - 4.0)
                        .abs()
                        < 1.0e-5
                );
                assert!((t.materialize_f64()[0] * t.materialize_f64()[1] - 1.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_preserves_row_vector_shape_for_callback() {
        let seen_shapes = Arc::new(Mutex::new(Vec::new()));
        let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let (x, shape) = match &args[0] {
                    Value::Tensor(t) => (t.materialize_f64().clone(), t.shape.clone()),
                    other => panic!("expected tensor input, got {other:?}"),
                };
                assert_eq!(shape, vec![1, 2]);
                seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![x[0] - 3.0, x[1] - 4.0], shape).unwrap(),
                    ))
                })
            },
        )));
        let x0 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("row_system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - 3.0).abs() < 1.0e-5);
                assert!((t.materialize_f64()[1] - 4.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
        assert!(!seen_shapes.lock().unwrap().is_empty());
    }

    #[test]
    fn fsolve_preserves_matrix_shape_for_callback() {
        let seen_shapes = Arc::new(Mutex::new(Vec::new()));
        let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let (x, shape) = match &args[0] {
                    Value::Tensor(t) => (t.materialize_f64().clone(), t.shape.clone()),
                    other => panic!("expected tensor input, got {other:?}"),
                };
                assert_eq!(shape, vec![2, 2]);
                seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![x[0] - 1.0, x[1] - 2.0, x[2] - 3.0, x[3] - 4.0], shape)
                            .unwrap(),
                    ))
                })
            },
        )));
        let x0 = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("matrix_system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1.0e-5);
                assert!((t.materialize_f64()[1] - 2.0).abs() < 1.0e-5);
                assert!((t.materialize_f64()[2] - 3.0).abs() < 1.0e-5);
                assert!((t.materialize_f64()[3] - 4.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
        assert!(!seen_shapes.lock().unwrap().is_empty());
    }

    #[test]
    fn fsolve_accepts_semantic_function_handle_callback() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 43);
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar numeric argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x - 3.0)) })
            },
        )));
        let root = block_on(fsolve_builtin(
            Value::BoundFunctionHandle {
                name: "system_function".to_string(),
                function: 43,
            },
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - 3.0).abs() < 1.0e-5),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FSOLVE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "x = fsolve(fun, x0)",
                "x = fsolve(fun, x0, options)",
                "[x,fval] = fsolve(fun, x0)",
                "[x,fval] = fsolve(fun, x0, options)",
                "[x,fval,exitflag] = fsolve(fun, x0)",
                "[x,fval,exitflag] = fsolve(fun, x0, options)",
                "[x,fval,exitflag,output] = fsolve(fun, x0)",
                "[x,fval,exitflag,output] = fsolve(fun, x0, options)",
                "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0)",
                "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0, options)",
            ]
        );

        let codes: Vec<&str> = FSOLVE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.FSOLVE.INVALID_ARGUMENT", "RM.FSOLVE.INVALID_INPUT"]
        );
    }

    #[test]
    fn fsolve_too_many_args_uses_stable_identifier() {
        let err = block_on(fsolve_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(1.0),
            vec![
                Value::Struct(StructValue::new()),
                Value::Struct(StructValue::new()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:fsolve:InvalidArgument"));
    }
}
