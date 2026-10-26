use crate::*;
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

const FZERO_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "fzero(fun, x0)",
        inputs: &FZERO_INPUTS_CORE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "fzero(fun, x0, options)",
        inputs: &FZERO_INPUTS_WITH_OPTIONS,
        outputs: &[],
    },
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

pub const FZERO_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FZERO.INVALID_ARGUMENT",
    identifier: Some("RunMat:fzero:InvalidArgument"),
    when: "Argument grammar/options struct are invalid.",
    message: "fzero: invalid argument",
};

pub const FZERO_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FZERO.INVALID_INPUT",
    identifier: Some("RunMat:fzero:InvalidInput"),
    when: "Callback/bracket/initial-point semantics are invalid.",
    message: "fzero: invalid input",
};

pub const FZERO_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
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

const FZERO_INTEGER_INITIAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x0 or [a,b]",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer initial points and intervals are independently gated and every value must convert exactly to binary64.",
    }];
const FZERO_INTEGER_CALLBACK_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fun(x) result",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer function values are gated before resident gather and must convert exactly to binary64.",
    }];
const FZERO_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "TolX",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolX is independently gated and must convert exactly to a positive binary64 scalar.",
    }];
const FZERO_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
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
pub const FZERO_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = fzero(fun, integer_x0_or_interval, options)",
        inputs: &FZERO_INTEGER_INITIAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented x0 supports single or double, not typed-integer classes. Strict compatibility rejects typed integers; RunMat mode admits exact binary64 values and returns double x, fval, and exitflag. A documented single x0 separately yields single numeric outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fzero callback returns an integer function value",
        inputs: &FZERO_INTEGER_CALLBACK_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Strict compatibility rejects typed function values before provider access. RunMat mode converts exact values at the binary64 solver boundary; numeric output precision remains determined by x0.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fzero(..., options.TolX=integer)",
        inputs: &FZERO_INTEGER_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed-integer TolX is RunMat-only. It overrides the precision-specific default without changing output precision.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fzero(..., options.MaxIter=integer, options.MaxFunEvals=integer)",
        inputs: &FZERO_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented controls are integer-valued numeric scalars, not documented typed-integer classes. RunMat's typed forms preserve exact counts without a binary64 round trip.",
    },
];

pub const FZERO_INPUT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fzero-nonfloating-initial-point",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fzero with a typed-integer or logical initial point or interval is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FzeroNumericInputExtension"),
};
pub const FZERO_CALLBACK_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fzero-nonfloating-callback-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fzero with typed-integer or logical function values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FzeroCallbackExtension"),
    };
pub const FZERO_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fzero-typed-option-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fzero with typed-integer option controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FzeroOptionExtension"),
};
pub const FZERO_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fzero-resident-fallback",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fzero with explicitly GPU-resident numeric input or callback output is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FzeroResidentExtension"),
};
pub const FZERO_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FZERO_INPUT_NUMERIC_EXTENSION,
    FZERO_CALLBACK_NUMERIC_EXTENSION,
    FZERO_OPTION_EXTENSION,
    FZERO_RESIDENT_EXTENSION,
];

pub const FZERO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FZERO_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FZERO_ERRORS,
};
