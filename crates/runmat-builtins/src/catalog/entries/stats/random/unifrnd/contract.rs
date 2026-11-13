//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const INTEGER_LOWER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-lower-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with a typed-integer lower bound is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerLowerBoundExtension"),
};

pub const INTEGER_UPPER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-upper-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with a typed-integer upper bound is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerUpperBoundExtension"),
};

pub const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerSizeExtension"),
};

pub const UNIFRND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_LOWER_BOUND_EXTENSION,
    INTEGER_UPPER_BOUND_EXTENSION,
    INTEGER_SIZE_EXTENSION,
];

pub const INTEGER_BOUND_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double lower bounds; typed integers cross a checked binary64 sampling boundary in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double upper bounds; typed integers cross a checked binary64 sampling boundary in RunMat mode.",
    },
];

pub const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single and double size controls; RunMat mode decodes typed integer extents exactly as structural values.",
}];

pub const UNIFRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unifrnd(integer_a, integer_b, ___)",
        inputs: &INTEGER_BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed bound is independently gated and must be exactly representable before entering the binary64 uniform generator.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unifrnd(a, b, integer_sz)",
        inputs: &INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed extents are extension-gated before provider access and parsed exactly without selecting the distribution computation domain.",
    },
];

pub const UNIFRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from uniform distribution.",
}];

pub const UNIFRND_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
];

pub const UNIFRND_INPUTS_A_B_SZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

pub const UNIFRND_INPUTS_A_B_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

pub const UNIFRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b)",
        inputs: &UNIFRND_INPUTS_A_B,
        outputs: &UNIFRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b, sz)",
        inputs: &UNIFRND_INPUTS_A_B_SZ,
        outputs: &UNIFRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b, sz1, sz2, ...)",
        inputs: &UNIFRND_INPUTS_A_B_DIMS,
        outputs: &UNIFRND_OUTPUT_R,
    },
];

pub const UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.UNIFRND.LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND",
        identifier: Some("RunMat:unifrnd:LowerBoundMustBeLessThanUpperBound"),
        when: "a is greater than or equal to b.",
        message: "unifrnd: a must be less than b",
    };

pub const UNIFRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIFRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:unifrnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "unifrnd: invalid argument",
};

pub const UNIFRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIFRND.INTERNAL",
    identifier: Some("RunMat:unifrnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "unifrnd: internal operation failed",
};

pub const UNIFRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND,
    UNIFRND_ERROR_INVALID_ARGUMENT,
    UNIFRND_ERROR_INTERNAL,
];

pub const UNIFRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIFRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNIFRND_ERRORS,
};
