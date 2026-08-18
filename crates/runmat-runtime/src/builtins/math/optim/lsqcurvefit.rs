//! MATLAB-compatible `lsqcurvefit` builtin for nonlinear curve fitting.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
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

const INTEGER_X0_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-x0",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with a native-class integer initial point is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerX0Extension"),
};
const INTEGER_XDATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-xdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with native-class integer model data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerXdataExtension"),
};
const INTEGER_YDATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-ydata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with native-class integer response data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerYdataExtension"),
};
const INTEGER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with native-class integer bounds is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerBoundExtension"),
};
const INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with native-class integer numeric options is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerOptionExtension"),
};
const INTEGER_CALLBACK_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-integer-callback-result",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with a native-class integer model result is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerCallbackExtension"),
};
const LOGICAL_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-logical-numeric",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit with logical solver data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitLogicalNumericExtension"),
};
const RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lsqcurvefit-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lsqcurvefit host fallback for explicit gpuArray values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LsqcurvefitResidentInputExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    INTEGER_X0_EXTENSION,
    INTEGER_XDATA_EXTENSION,
    INTEGER_YDATA_EXTENSION,
    INTEGER_BOUND_EXTENSION,
    INTEGER_OPTION_EXTENSION,
    INTEGER_CALLBACK_EXTENSION,
    LOGICAL_NUMERIC_EXTENSION,
    RESIDENT_INPUT_EXTENSION,
];

const INTEGER_X0_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x0",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer initial points are gated before gather and cross only an exact binary64 solver boundary.",
}];
const INTEGER_XDATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "xdata",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer model data remains exact callback payload rather than being eagerly materialized as double.",
}];
const INTEGER_YDATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ydata or fun result",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer response and model values are independently gated and must be exact at residual subtraction.",
}];
const INTEGER_BOUND_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "lb or ub",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer bounds are gated before gather and must be exactly representable in the double solver domain.",
}];
const INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "numeric option field",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed MaxIter and MaxFunEvals are parsed structurally; typed tolerances cross a checked binary64 boundary.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor { form: "x = lsqcurvefit(fun, integer_x0, xdata, ydata, ___)", inputs: &INTEGER_X0_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The host nonlinear solver works in double and rejects any integer initial point that cannot cross exactly." },
    BuiltinIntegerCapabilityDescriptor { form: "x = lsqcurvefit(fun, x0, integer_xdata, ydata, ___)", inputs: &INTEGER_XDATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat passes exact integer model data to the callback unchanged, including supported resident payloads." },
    BuiltinIntegerCapabilityDescriptor { form: "x = lsqcurvefit(fun, x0, xdata, integer_ydata, ___)", inputs: &INTEGER_YDATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Response data and callback results enter double residual arithmetic only after exact representability checks." },
    BuiltinIntegerCapabilityDescriptor { form: "x = lsqcurvefit(fun, x0, xdata, ydata, integer_lb, integer_ub, ___)", inputs: &INTEGER_BOUND_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer bounds are a RunMat-only extension and cross the double bound projection boundary exactly." },
    BuiltinIntegerCapabilityDescriptor { form: "x = lsqcurvefit(___, options_with_integer_field)", inputs: &INTEGER_OPTION_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Integer option fields are gated before recursive gather; counts remain exact structural controls and tolerances convert exactly." },
];

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
    extensions(crate::builtins::math::optim::lsqcurvefit::EXTENSIONS),
    integer_capabilities(crate::builtins::math::optim::lsqcurvefit::INTEGER_CAPABILITIES),
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
    ensure_input_extensions(&x0, &xdata, &ydata, &rest)?;
    let parsed = ParsedArgs::parse(rest)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let parsed = parsed
        .prepare_options()
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_ARGUMENT))?;
    let x0 = prepare_floating_input("initial point", x0, &INTEGER_X0_EXTENSION)
        .await
        .map_err(|err| map_error(err, &ERROR_INVALID_INPUT))?;
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
    let ydata = real_array_checked("ydata", ydata, &INTEGER_YDATA_EXTENSION)
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

fn ensure_input_extensions(
    x0: &Value,
    xdata: &Value,
    ydata: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        value_contains_explicit_gpu, value_contains_native_integer_class,
    };
    if value_contains_native_integer_class(x0) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X0_EXTENSION, NAME)?;
    }
    if value_contains_native_integer_class(xdata) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_XDATA_EXTENSION, NAME)?;
    }
    if value_contains_native_integer_class(ydata) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_YDATA_EXTENSION, NAME)?;
    }
    if rest.len() >= 2 {
        for value in rest.iter().take(2) {
            if value_contains_native_integer_class(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_BOUND_EXTENSION,
                    NAME,
                )?;
            }
        }
    }
    if is_logical_numeric(x0) || is_logical_numeric(xdata) || is_logical_numeric(ydata) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_NUMERIC_EXTENSION, NAME)?;
    }
    if value_contains_explicit_gpu(x0)
        || value_contains_explicit_gpu(xdata)
        || value_contains_explicit_gpu(ydata)
        || rest.iter().any(value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_INPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

fn is_logical_numeric(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

async fn prepare_floating_input(
    role: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, NAME)?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(lsq_error_with_detail(
                &ERROR_INVALID_ARGUMENT,
                format!("integer {role} must be exactly representable as double"),
            ));
        }
    }
    if is_logical_numeric(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_NUMERIC_EXTENSION, NAME)?;
    }
    crate::dispatcher::gather_if_needed_async(&value).await
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

    async fn prepare_options(mut self) -> BuiltinResult<Self> {
        let Some(options) = self.options.take() else {
            return Ok(self);
        };
        for field in ["TolX", "TolFun", "MaxIter", "MaxFunEvals"] {
            let Some(value) = options
                .fields
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(field))
                .map(|(_, value)| value)
            else {
                continue;
            };
            if crate::builtins::common::validation::value_contains_native_integer_class(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_OPTION_EXTENSION,
                    NAME,
                )?;
                if matches!(field, "TolX" | "TolFun")
                    && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(
                        value,
                    )
                    .await?
                {
                    return Err(lsq_error_with_detail(
                        &ERROR_INVALID_ARGUMENT,
                        format!("integer option {field} must be exactly representable as double"),
                    ));
                }
            }
        }
        let gathered = crate::dispatcher::gather_if_needed_async(&Value::Struct(options)).await?;
        let Value::Struct(options) = gathered else {
            unreachable!("gather preserves struct shape")
        };
        self.options = Some(options);
        Ok(self)
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
            let model =
                real_array_checked("model output", value, &INTEGER_CALLBACK_EXTENSION).await?;
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

async fn real_array_checked(
    label: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<RealArray> {
    let value = prepare_floating_input(label, value, integer_extension).await?;
    real_array(label, value).await
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
    let value = prepare_floating_input(label, value, &INTEGER_BOUND_EXTENSION).await?;
    if is_empty_value(&value) {
        return Ok(vec![default; n]);
    }
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(flag) => vec![if flag { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => tensor::tensor_into_values_f64(tensor),
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
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
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
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntegerStorage;
    use std::sync::Arc;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn lsqcurvefit_real_array_reads_typed_integer_storage_exactly() {
        let input =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3]).expect("integer");

        let parsed = block_on(real_array("xdata", Value::Tensor(input))).expect("real array");

        assert_eq!(parsed.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(parsed.shape, vec![1, 3]);
    }

    #[test]
    fn lsqcurvefit_bound_vector_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![-1, 2]), vec![1, 2]).expect("integer");

        let parsed = block_on(bound_vector(
            "lower bounds",
            Value::Tensor(input),
            2,
            f64::NEG_INFINITY,
        ))
        .expect("bounds");

        assert_eq!(parsed, vec![-1.0, 2.0]);
    }

    #[test]
    fn lsqcurvefit_strict_mode_rejects_integer_initial_point() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let x0 = Tensor::new_integer(IntegerStorage::I32(vec![0]), vec![1, 1]).unwrap();

        let error = block_on(lsqcurvefit_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(x0),
            Value::Num(0.0),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect_err("integer x0 is a RunMat-only extension");

        assert_eq!(error.identifier(), INTEGER_X0_EXTENSION.error_identifier);
    }

    #[test]
    fn lsqcurvefit_runmat_mode_rejects_wide_integer_initial_point() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let x0 =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();

        let error = block_on(lsqcurvefit_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(x0),
            Value::Num(0.0),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect_err("wide integer x0 cannot cross exactly");

        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn lsqcurvefit_passes_wide_integer_xdata_to_callback_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let expected = u64::MAX;
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let Value::Tensor(xdata) = &args[1] else {
                    panic!("expected exact integer xdata")
                };
                assert!(matches!(
                    xdata.numeric_value_at(0),
                    Some(runmat_builtins::NumericScalar::U64(value)) if value == expected
                ));
                Box::pin(async move { Ok(Value::Num(0.0)) })
            },
        )));
        let xdata = Tensor::new_integer(IntegerStorage::U64(vec![expected]), vec![1, 1]).unwrap();

        let result = block_on(lsqcurvefit_builtin(
            Value::BoundFunctionHandle {
                name: "wide_xdata".to_string(),
                function: 901,
            },
            Value::Num(0.0),
            Value::Tensor(xdata),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect("exact xdata pass-through");

        assert!(matches!(result, Value::Num(value) if value == 0.0));
    }

    #[test]
    fn lsqcurvefit_automatic_resident_input_gathers_but_explicit_input_is_gated() {
        test_support::with_test_provider(|provider| {
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(|_function, _args, _requested_outputs| {
                    Box::pin(async move { Ok(Value::Num(0.0)) })
                }),
            ));
            let values = [0.0];
            let shape = [1, 1];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic =
                automatic.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let result = block_on(lsqcurvefit_builtin(
                Value::BoundFunctionHandle {
                    name: "zero_model".to_string(),
                    function: 904,
                },
                Value::GpuTensor(automatic),
                Value::Num(0.0),
                Value::Num(0.0),
                Vec::new(),
            ))
            .expect("automatic resident initial point gathers");
            assert!(matches!(
                result,
                Value::Tensor(ref tensor)
                    if tensor.shape == vec![1, 1]
                        && matches!(tensor.numeric_value_at(0), Some(runmat_builtins::NumericScalar::F64(value)) if value == 0.0)
            ));

            let explicit = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit =
                explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(lsqcurvefit_builtin(
                Value::FunctionHandle("unused".into()),
                Value::GpuTensor(explicit),
                Value::Num(0.0),
                Value::Num(0.0),
                Vec::new(),
            ))
            .expect_err("explicit resident input is gated before fallback");
            assert_eq!(
                error.identifier(),
                RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn lsqcurvefit_linear_model_recovers_parameters() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let p = match &args[0] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
                assert!((t.materialize_f64()[0] - 2.0).abs() < 1.0e-5);
                assert!((t.materialize_f64()[1] - 1.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_exponential_model_recovers_parameters() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
                assert!((t.materialize_f64()[0] - 2.5).abs() < 1.0e-4);
                assert!((t.materialize_f64()[1] - 0.7).abs() < 1.0e-4);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn lsqcurvefit_respects_bounds() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let p = match &args[0] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
            Value::Tensor(t) => assert!((t.materialize_f64()[0] - 1.0).abs() < 1.0e-8),
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
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    other => panic!("expected params, got {other:?}"),
                };
                let xdata = match &args[1] {
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
                        for (actual, expected) in j.materialize_f64().iter().zip(expected) {
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
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
                    Value::Tensor(t) => t.materialize_f64().clone(),
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
            Value::Tensor(t) => {
                assert!((t.materialize_f64()[0] + t.materialize_f64()[1] - 3.0).abs() < 1.0e-6)
            }
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
