use crate::*;

pub const LSQCURVEFIT_INTEGER_X0_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-x0",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with a native-class integer initial point is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerX0Extension"),
    };
pub const LSQCURVEFIT_INTEGER_XDATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-xdata",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with native-class integer model data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerXdataExtension"),
    };
pub const LSQCURVEFIT_INTEGER_YDATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-ydata",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with native-class integer response data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerYdataExtension"),
    };
pub const LSQCURVEFIT_INTEGER_BOUND_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-bound",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with native-class integer bounds is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerBoundExtension"),
    };
pub const LSQCURVEFIT_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with native-class integer numeric options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerOptionExtension"),
    };
pub const LSQCURVEFIT_INTEGER_CALLBACK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-integer-callback-result",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with a native-class integer model result is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitIntegerCallbackExtension"),
    };
pub const LSQCURVEFIT_LOGICAL_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-logical-numeric",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit with logical solver data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitLogicalNumericExtension"),
    };
pub const LSQCURVEFIT_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "lsqcurvefit-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "lsqcurvefit host fallback for explicit gpuArray values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LsqcurvefitResidentInputExtension"),
    };

pub const LSQCURVEFIT_EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    LSQCURVEFIT_INTEGER_X0_EXTENSION,
    LSQCURVEFIT_INTEGER_XDATA_EXTENSION,
    LSQCURVEFIT_INTEGER_YDATA_EXTENSION,
    LSQCURVEFIT_INTEGER_BOUND_EXTENSION,
    LSQCURVEFIT_INTEGER_OPTION_EXTENSION,
    LSQCURVEFIT_INTEGER_CALLBACK_EXTENSION,
    LSQCURVEFIT_LOGICAL_NUMERIC_EXTENSION,
    LSQCURVEFIT_RESIDENT_INPUT_EXTENSION,
];

const INTEGER_X0_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x0",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer initial points are gated before gather and cross only an exact binary64 solver boundary.",
}];
const INTEGER_XDATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "xdata",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer model data remains exact callback payload rather than being eagerly materialized as double.",
}];
const INTEGER_YDATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ydata or fun result",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer response and model values are independently gated and must be exact at residual subtraction.",
}];
const INTEGER_BOUND_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "lb or ub",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer bounds are gated before gather and must be exactly representable in the double solver domain.",
}];
const INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "numeric option field",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed MaxIter and MaxFunEvals are parsed structurally; typed tolerances cross a checked binary64 boundary.",
}];

pub const LSQCURVEFIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
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

const INPUTS_OPTIONS: [BuiltinParamDescriptor; 5] = [
    input_fun(),
    input_x0(),
    input_xdata(),
    input_ydata(),
    input_options(),
];

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

const SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "x = lsqcurvefit(fun, x0, xdata, ydata)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqcurvefit(fun, x0, xdata, ydata, options)",
        inputs: &INPUTS_OPTIONS,
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

pub const LSQCURVEFIT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqcurvefit:InvalidArgument"),
    when: "Argument grammar, bounds, options, or output arity are invalid.",
    message: "lsqcurvefit: invalid argument",
};

pub const LSQCURVEFIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.INVALID_INPUT",
    identifier: Some("RunMat:lsqcurvefit:InvalidInput"),
    when: "Initial guess, model callback, data shape, or solver semantics are invalid.",
    message: "lsqcurvefit: invalid input",
};

pub const LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQCURVEFIT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:lsqcurvefit:TooManyOutputs"),
    when: "`lsqcurvefit` is called with more than seven requested outputs.",
    message: "lsqcurvefit: too many output arguments",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
    LSQCURVEFIT_ERROR_INVALID_INPUT,
    LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS,
];

pub const LSQCURVEFIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
