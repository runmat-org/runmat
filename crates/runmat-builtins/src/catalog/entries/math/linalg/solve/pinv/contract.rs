use crate::*;

pub const PINV_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pinv-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pinv with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PinvIntegerInputExtension"),
};

pub const PINV_INTEGER_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pinv-integer-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pinv with a typed-integer tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PinvIntegerToleranceExtension"),
    };
pub const PINV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pinv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pinv with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PinvLogicalInputExtension"),
};
pub const PINV_LOGICAL_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pinv-logical-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pinv with a logical tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PinvLogicalToleranceExtension"),
    };

pub const PINV_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    PINV_INTEGER_INPUT_EXTENSION,
    PINV_INTEGER_TOLERANCE_EXTENSION,
    PINV_LOGICAL_INPUT_EXTENSION,
    PINV_LOGICAL_TOLERANCE_EXTENSION,
];

pub const PINV_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public matrix classes are single and double; RunMat mode admits real integer matrices only when every value is exactly representable as binary64.",
    }];

pub const PINV_INTEGER_TOLERANCE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "tol",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public tolerance is a real scalar without an integer class list; RunMat mode admits typed integer tolerance values only when exactly representable as binary64.",
    }];

pub const PINV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = pinv(integer_A, tol?)",
        inputs: &PINV_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated integer matrix crosses one checked binary64 SVD boundary. Host output is double, and automatic residency may restore the floating result to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = pinv(A, integer_tol)",
        inputs: &PINV_INTEGER_TOLERANCE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer tolerance is a gated RunMat extension and crosses into the SVD threshold only after exact binary64 representability and nonnegative finite validation.",
    },
];

pub const PINV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pseudoinverse of A.",
}];

pub const PINV_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const PINV_INPUTS_TOL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Singular-value threshold.",
    },
];

pub const PINV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = pinv(A)",
        inputs: &PINV_INPUTS,
        outputs: &PINV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "X = pinv(A, tol)",
        inputs: &PINV_INPUTS_TOL,
        outputs: &PINV_OUTPUT,
    },
];

pub const PINV_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PINV.INVALID_ARGUMENT",
    identifier: Some("RunMat:pinv:InvalidArgument"),
    when: "Optional tolerance argument is malformed or outside accepted bounds.",
    message: "pinv: invalid argument",
};

pub const PINV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PINV.INVALID_INPUT",
    identifier: Some("RunMat:pinv:InvalidInput"),
    when: "Input shape/type cannot be processed for pseudoinverse evaluation.",
    message: "pinv: invalid input",
};

pub const PINV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PINV.INTERNAL",
    identifier: Some("RunMat:pinv:Internal"),
    when: "Runtime fails while computing pseudoinverse or executing fallback/upload paths.",
    message: "pinv: internal runtime failure",
};

pub const PINV_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PINV_ERROR_INVALID_ARGUMENT,
    PINV_ERROR_INVALID_INPUT,
    PINV_ERROR_INTERNAL,
];

pub const PINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PINV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PINV_ERRORS,
};
