use crate::*;

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
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "QN approximate Hessian ([] for L-BFGS), last constructed TR model or original Hinfo, or [] when no TR model was constructed.",
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

pub const FMINUNC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINUNC.INVALID_ARGUMENT",
    identifier: Some("RunMat:fminunc:InvalidArgument"),
    when: "Argument grammar/options parsing is invalid.",
    message: "fminunc: invalid argument",
};

pub const FMINUNC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINUNC.INVALID_INPUT",
    identifier: Some("RunMat:fminunc:InvalidInput"),
    when: "Initial guess/callback/iteration semantics are invalid.",
    message: "fminunc: invalid input",
};

const FMINUNC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FMINUNC_ERROR_INVALID_ARGUMENT, FMINUNC_ERROR_INVALID_INPUT];

const FMINUNC_INTEGER_INITIAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x0",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer initial guesses are independently gated and every element must convert exactly to binary64.",
    }];
const FMINUNC_INTEGER_CALLBACK_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "objective result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer objective values are gated before resident gather and must convert exactly to binary64.",
    },
    BuiltinIntegerInputCapability {
        name: "gradient result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer supplied gradients are gated independently and every element must convert exactly to binary64.",
    },
    BuiltinIntegerInputCapability {
        name: "Hessian result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer objective Hessians are independently gated and must convert exactly to binary64 before matrix-operator admission.",
    },
    BuiltinIntegerInputCapability {
        name: "Hessian product result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer multiply outputs are independently gated and each value must convert exactly to binary64. Arbitrary Hinfo is not coerced or walked.",
    },

];
const FMINUNC_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 5] = [
    BuiltinIntegerInputCapability {
        name: "TolX",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolX is independently gated and must convert exactly to a nonnegative binary64 scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "TolFun",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolFun is independently gated and must convert exactly to a nonnegative binary64 scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "OptimalityTolerance",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer tolerances are gated and must convert exactly to a nonnegative finite double.",
    },
    BuiltinIntegerInputCapability {
        name: "FunctionTolerance",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The active TR function-change tolerance is gated and must convert exactly to a nonnegative finite double.",
    },
    BuiltinIntegerInputCapability {
        name: "TolPCG",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Active typed-integer PCG tolerances are gated and must convert exactly to a positive finite double.",
    },

];
const FMINUNC_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "MaxIter",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer iteration counts are independently gated and decoded exactly through the solver limit.",
    },
    BuiltinIntegerInputCapability {
        name: "MaxFunEvals",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer evaluation counts are independently gated and decoded exactly through the solver limit.",
    },
    BuiltinIntegerInputCapability {
        name: "MaxPCGIter",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Active typed-integer PCG limits are gated and decoded exactly as positive structural counts.",
    },
    BuiltinIntegerInputCapability {
        name: "PrecondBandWidth",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "An active finite typed-integer bandwidth is gated and decoded exactly as a nonnegative structural count. Infinity is only the floating direct-solve spelling.",
    },

];
const FMINUNC_INTEGER_PATTERN_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {name:"HessPattern",classes:&crate::catalog::integer::ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,notes:"Active typed-integer structural patterns are extension-gated; zero/nonzero structure is read without requiring a lossy numeric derivative conversion."}];
const FMINUNC_INTEGER_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "SpecifyObjectiveGradient",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer 0 or 1 is a RunMat-only spelling. Documented logical false or true remains accepted in strict compatibility mode.",
    }];
pub const FMINUNC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = fminunc(fun, integer_x0, options)",
        inputs: &FMINUNC_INTEGER_INITIAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented x0 is double. Strict compatibility rejects typed integers; RunMat mode admits exact binary64 values and returns the ordinary double x, fval, gradient, and Hessian outputs plus diagnostics.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminunc callback returns integer objective or derivative data",
        inputs: &FMINUNC_INTEGER_CALLBACK_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Strict compatibility rejects typed numeric callback outputs before provider access; opaque Hinfo is preserved as supplied. RunMat mode converts exact values to binary64; solver outputs retain their function-specific double and diagnostic classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminunc(..., typed-integer step, optimality, function-change or PCG tolerance controls)",
        inputs: &FMINUNC_INTEGER_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer tolerance controls are RunMat-only and do not change output classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminunc(..., typed-integer iteration, evaluation, PCG or finite-bandwidth controls)",
        inputs: &FMINUNC_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented controls are integer-valued numeric scalars, not documented typed-integer classes. RunMat's typed forms preserve exact counts without a binary64 round trip.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminunc(..., options.SpecifyObjectiveGradient=integer)",
        inputs: &FMINUNC_INTEGER_FLAG_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only typed-integer 0 or 1 uses this extension. Logical option values are documented and bypass the typed-integer extension gate.",
    },
    BuiltinIntegerCapabilityDescriptor {form:"fminunc(..., typed-integer HessPattern)",inputs:&FMINUNC_INTEGER_PATTERN_INPUTS,computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Only an active finite-difference pattern is consumed. Numeric values encode structural zero/nonzero positions, not Hessian coefficients; logical masks remain standard."},
];

pub const FMINUNC_INPUT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fminunc-nonfloating-initial-point",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fminunc with a typed-integer or logical initial point is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FminuncNumericInputExtension"),
    };
pub const FMINUNC_CALLBACK_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminunc-nonfloating-callback-output", mode: BuiltinExtensionMode::RunMatOnly,
    description: "fminunc with typed-integer or logical objective or derivative output is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FminuncCallbackExtension"),
};
pub const FMINUNC_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminunc-typed-option-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fminunc with typed-integer option controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FminuncOptionExtension"),
};
pub const FMINUNC_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminunc-resident-fallback",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fminunc with explicitly GPU-resident numeric input or callback output is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FminuncResidentExtension"),
};
pub const FMINUNC_NUMERICAL_FAILURE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminunc-numerical-failure-status",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fminunc exitflag -4 returns the last accepted point on numerical line-search or trust-region subproblem failure, without claiming a MATLAB exit status or a local minimum",
    error_identifier: Some("RunMat:compatibility:FminuncNumericalFailureExtension"),
};
pub const FMINUNC_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    FMINUNC_INPUT_NUMERIC_EXTENSION,
    FMINUNC_CALLBACK_NUMERIC_EXTENSION,
    FMINUNC_OPTION_EXTENSION,
    FMINUNC_RESIDENT_EXTENSION,
    FMINUNC_NUMERICAL_FAILURE_EXTENSION,
];

pub const FMINUNC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FMINUNC_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FMINUNC_ERRORS,
};
