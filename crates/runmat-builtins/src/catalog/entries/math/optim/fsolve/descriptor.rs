use crate::*;

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

pub const FSOLVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSOLVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:fsolve:InvalidArgument"),
    when: "Argument grammar/options configuration is invalid.",
    message: "fsolve: invalid argument",
};

pub const FSOLVE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
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
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer initial guesses are independently gated and every element must convert exactly to binary64.",
    }];
const FSOLVE_INTEGER_RESIDUAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fun(x) residual",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer residuals are gated before resident gather and every element must convert exactly to binary64.",
    }];
const FSOLVE_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "TolX",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolX is independently gated and must convert exactly to a positive binary64 scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "TolFun",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer TolFun is independently gated and must convert exactly to a positive binary64 scalar.",
    },
];
const FSOLVE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
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

pub const FSOLVE_INPUT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsolve-nonfloating-initial-point",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsolve with a typed-integer or logical initial point is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsolveNumericInputExtension"),
};
pub const FSOLVE_CALLBACK_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsolve-nonfloating-callback-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsolve with typed-integer or logical residual output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsolveCallbackExtension"),
    };
pub const FSOLVE_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsolve-typed-option-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsolve with typed-integer option controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsolveOptionExtension"),
};
pub const FSOLVE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsolve-resident-fallback",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fsolve with explicitly GPU-resident numeric input or callback output is a RunMat extension",
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
