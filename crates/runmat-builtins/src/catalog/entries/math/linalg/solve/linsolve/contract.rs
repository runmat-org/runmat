use crate::*;

pub const LINSOLVE_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solution to A * X = B.",
}];

pub const LINSOLVE_OUTPUT_XR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solution to A * X = B.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reciprocal condition estimate.",
    },
];

pub const LINSOLVE_INPUTS_AB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Coefficient matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right-hand side matrix or vector.",
    },
];

pub const LINSOLVE_INPUTS_AB_OPTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Coefficient matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right-hand side matrix or vector.",
    },
    BuiltinParamDescriptor {
        name: "opts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Structural options (LT, UT, RECT, SYM, POSDEF, TRANSA, RCOND).",
    },
];

pub const LINSOLVE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "X = linsolve(A, B)",
        inputs: &LINSOLVE_INPUTS_AB,
        outputs: &LINSOLVE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = linsolve(A, B, opts)",
        inputs: &LINSOLVE_INPUTS_AB_OPTS,
        outputs: &LINSOLVE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X, R] = linsolve(A, B)",
        inputs: &LINSOLVE_INPUTS_AB,
        outputs: &LINSOLVE_OUTPUT_XR,
    },
    BuiltinSignatureDescriptor {
        label: "[X, R] = linsolve(A, B, opts)",
        inputs: &LINSOLVE_INPUTS_AB_OPTS,
        outputs: &LINSOLVE_OUTPUT_XR,
    },
];

pub const LINSOLVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSOLVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:linsolve:InvalidArgument"),
    when: "Options/output count/auxiliary arguments are malformed or unsupported.",
    message: "linsolve: invalid argument",
};

pub const LINSOLVE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSOLVE.INVALID_INPUT",
    identifier: Some("RunMat:linsolve:InvalidInput"),
    when: "Input shape/type cannot be solved under linsolve semantics.",
    message: "linsolve: invalid input",
};

pub const LINSOLVE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSOLVE.INTERNAL",
    identifier: Some("RunMat:linsolve:Internal"),
    when: "Runtime fails while solving or executing provider fallback paths.",
    message: "linsolve: internal runtime failure",
};

pub const LINSOLVE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LINSOLVE_ERROR_INVALID_ARGUMENT,
    LINSOLVE_ERROR_INVALID_INPUT,
    LINSOLVE_ERROR_INTERNAL,
];
pub const LINSOLVE_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linsolve-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linsolve with integer A or B is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinsolveIntegerInputExtension"),
    };
pub const LINSOLVE_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linsolve-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linsolve with logical A or B is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinsolveLogicalInputExtension"),
    };
pub const LINSOLVE_EXPLICIT_GPU_TWO_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linsolve-explicit-gpu-two-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "two-output linsolve with explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinsolveExplicitGpuTwoOutputExtension"),
    };
pub const LINSOLVE_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linsolve-integer-option-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linsolve with a typed-integer structural option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinsolveIntegerOptionExtension"),
    };
pub const LINSOLVE_TEXT_TRANSA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linsolve-text-transa-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linsolve with a text-valued TRANSA option is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinsolveTextTransaExtension"),
};
pub const LINSOLVE_RCOND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linsolve-rcond-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linsolve with the RCOND option is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinsolveRcondExtension"),
};
pub const LINSOLVE_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    LINSOLVE_INTEGER_INPUT_EXTENSION,
    LINSOLVE_LOGICAL_INPUT_EXTENSION,
    LINSOLVE_EXPLICIT_GPU_TWO_OUTPUT_EXTENSION,
    LINSOLVE_INTEGER_OPTION_EXTENSION,
    LINSOLVE_TEXT_TRANSA_EXTENSION,
    LINSOLVE_RCOND_EXTENSION,
];
pub const LINSOLVE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only exact-owner promotion into an exact binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only exact-owner promotion into an exact binary64 boundary.",
    },
];
pub const LINSOLVE_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "opts structural fields",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer LT, UT, RECT, SYM, and POSDEF values are independently gated structural controls.",
    }];
pub const LINSOLVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "X = linsolve(integer_A, integer_B)", inputs: &LINSOLVE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer operands are a gated RunMat extension and are promoted only when exactly representable in binary64." },
    BuiltinIntegerCapabilityDescriptor { form: "X = linsolve(A, B, opts_with_integer_field)", inputs: &LINSOLVE_INTEGER_OPTION_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "RunMat-only typed-integer truth controls are classified before option coercion." },
];

pub const LINSOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINSOLVE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINSOLVE_ERRORS,
};
