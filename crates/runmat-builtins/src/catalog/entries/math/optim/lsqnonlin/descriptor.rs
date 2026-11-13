use crate::*;

const INTEGER_DATA: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "x0, bounds or residual callback result",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Existing RunMat forms normalize native integer solver data into binary64, including rounding outside its exact integer range. This is not an exact-integer solver or a claim of MATLAB integer-overload parity.",
}];
const INTEGER_COUNTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "options.MaxIter or options.MaxFunEvals",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Structural native integer counts use exact checked platform usize decoding. Zero iterations are honored. Zero function evaluations stop before the first callback and return an explicitly unevaluated supplied candidate with unknown numerical products.",
}];
const INTEGER_TOLERANCES: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "options.TolX or options.TolFun",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer tolerance values must be exactly representable at the shared finite binary64 option boundary.",
}];
pub const LSQNONLIN_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form: "lsqnonlin(integer solver data)", inputs: INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple, notes: "Preserves the existing floating solver boundary without adding a new rejection or exactness claim." },
    BuiltinIntegerCapabilityDescriptor { form: "lsqnonlin(integer count options)", inputs: INTEGER_COUNTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Exact structural counts do not change the binary64 solution and diagnostic classes." },
    BuiltinIntegerCapabilityDescriptor { form: "lsqnonlin(integer tolerances)", inputs: INTEGER_TOLERANCES,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Tolerance validation follows the existing shared option owner." },
];

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
        description: "Real Jacobian at the accepted solution: supplied, finite-differenced, or formed through opaque forward products when requested.",
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

const fn constraint_placeholder(name: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("[]"),
        description: "Only an empty constraint placeholder is supported.",
    }
}
const INPUTS_LINEAR_PLACEHOLDERS: [BuiltinParamDescriptor; 8] = [
    input_fun(),
    input_x0(),
    input_lb(),
    input_ub(),
    constraint_placeholder("A"),
    constraint_placeholder("b"),
    constraint_placeholder("Aeq"),
    constraint_placeholder("beq"),
];
const INPUTS_CONSTRAINT_PLACEHOLDERS: [BuiltinParamDescriptor; 9] = [
    input_fun(),
    input_x0(),
    input_lb(),
    input_ub(),
    constraint_placeholder("A"),
    constraint_placeholder("b"),
    constraint_placeholder("Aeq"),
    constraint_placeholder("beq"),
    constraint_placeholder("nonlcon"),
];
const INPUTS_CONSTRAINT_OPTIONS: [BuiltinParamDescriptor; 10] = [
    input_fun(),
    input_x0(),
    input_lb(),
    input_ub(),
    constraint_placeholder("A"),
    constraint_placeholder("b"),
    constraint_placeholder("Aeq"),
    constraint_placeholder("beq"),
    constraint_placeholder("nonlcon"),
    input_options(),
];

const SIGNATURES: [BuiltinSignatureDescriptor; 13] = [
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
        label: "x = lsqnonlin(fun, x0, lb, ub, [], [], [], [])",
        inputs: &INPUTS_LINEAR_PLACEHOLDERS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0, lb, ub, [], [], [], [], [])",
        inputs: &INPUTS_CONSTRAINT_PLACEHOLDERS,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lsqnonlin(fun, x0, lb, ub, [], [], [], [], [], options)",
        inputs: &INPUTS_CONSTRAINT_OPTIONS,
        outputs: &OUTPUT_X,
    },
];

pub const LSQNONLIN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqnonlin:InvalidArgument"),
    when: "Argument grammar, bounds, options, constraints, problem struct, or output arity are invalid.",
    message: "lsqnonlin: invalid argument",
};

pub const LSQNONLIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.INVALID_INPUT",
    identifier: Some("RunMat:lsqnonlin:InvalidInput"),
    when: "Initial guess, residual callback, or solver semantics are invalid.",
    message: "lsqnonlin: invalid input",
};

pub const LSQNONLIN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONLIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:lsqnonlin:TooManyOutputs"),
    when: "`lsqnonlin` is called with more than seven requested outputs.",
    message: "lsqnonlin: too many output arguments",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LSQNONLIN_ERROR_INVALID_ARGUMENT,
    LSQNONLIN_ERROR_INVALID_INPUT,
    LSQNONLIN_ERROR_TOO_MANY_OUTPUTS,
];

pub const LSQNONLIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
