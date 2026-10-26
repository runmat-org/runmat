use crate::*;
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

const FMINBND_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "fminbnd(fun, x1, x2)",
        inputs: &FMINBND_INPUTS_CORE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "fminbnd(fun, x1, x2, options)",
        inputs: &FMINBND_INPUTS_WITH_OPTIONS,
        outputs: &[],
    },
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

pub const FMINBND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINBND.INVALID_ARGUMENT",
    identifier: Some("RunMat:fminbnd:InvalidArgument"),
    when: "Argument grammar/options parsing is invalid.",
    message: "fminbnd: invalid argument",
};

pub const FMINBND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINBND.INVALID_INPUT",
    identifier: Some("RunMat:fminbnd:InvalidInput"),
    when: "Bounds/callback/input scalar semantics are invalid.",
    message: "fminbnd: invalid input",
};

pub const FMINBND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINBND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fminbnd:TooManyOutputs"),
    when: "More than four output arguments are requested.",
    message: "fminbnd: too many output arguments",
};
const FMINBND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FMINBND_ERROR_INVALID_ARGUMENT,
    FMINBND_ERROR_INVALID_INPUT,
    FMINBND_ERROR_TOO_MANY_OUTPUTS,
];

const FMINBND_INTEGER_BOUND_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x1",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer lower bounds are an independently gated RunMat extension and must convert exactly to binary64.",
    },
    BuiltinIntegerInputCapability {
        name: "x2",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer upper bounds are an independently gated RunMat extension and must convert exactly to binary64.",
    },
];
const FMINBND_INTEGER_OBJECTIVE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fun(x) result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer objective results are gated before resident gather and must convert exactly to binary64.",
    }];
const FMINBND_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "TolX",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolX is independently gated and must convert exactly to a positive binary64 scalar.",
    }];
const FMINBND_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "MaxIter",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer iteration counts are independently gated and decoded exactly through platform bounds.",
    },
    BuiltinIntegerInputCapability {
        name: "MaxFunEvals",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer evaluation counts are independently gated and decoded exactly through platform bounds.",
    },
];
pub const FMINBND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = fminbnd(fun, integer_x1, integer_x2, options)",
        inputs: &FMINBND_INTEGER_BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Documented bounds are double. Strict compatibility rejects typed integers; RunMat mode admits exact binary64 values and returns the ordinary double solver outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = fminbnd(fun_returning_integer, x1, x2, options)",
        inputs: &FMINBND_INTEGER_OBJECTIVE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Strict compatibility rejects typed-integer objective values before provider access; RunMat mode converts exact values to binary64 and returns double x and fval plus function-specific diagnostics.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminbnd(..., options.TolX=integer)",
        inputs: &FMINBND_INTEGER_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer tolerance controls are RunMat-only and do not change the documented output classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fminbnd(..., options.MaxIter=integer, options.MaxFunEvals=integer)",
        inputs: &FMINBND_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented controls are integer-valued numeric scalars, not documented typed-integer classes. RunMat's typed forms preserve exact counts without a binary64 round trip.",
    },
];

pub const FMINBND_INPUT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fminbnd-nonfloating-bounds",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fminbnd with typed-integer or logical bounds is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FminbndNumericInputExtension"),
    };
pub const FMINBND_CALLBACK_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fminbnd-nonfloating-objective",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fminbnd with typed-integer or logical objective values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FminbndCallbackExtension"),
    };
pub const FMINBND_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminbnd-typed-option-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fminbnd with typed-integer option controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FminbndOptionExtension"),
};
pub const FMINBND_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fminbnd-resident-fallback",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fminbnd with explicitly GPU-resident numeric input or callback output is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FminbndResidentExtension"),
};
pub const FMINBND_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FMINBND_INPUT_NUMERIC_EXTENSION,
    FMINBND_CALLBACK_NUMERIC_EXTENSION,
    FMINBND_OPTION_EXTENSION,
    FMINBND_RESIDENT_EXTENSION,
];

pub const FMINBND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FMINBND_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FMINBND_ERRORS,
};
