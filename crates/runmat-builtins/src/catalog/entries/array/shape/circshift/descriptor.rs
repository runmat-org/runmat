use crate::*;

const CIRCSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with elements circularly shifted.",
}];

const CIRCSHIFT_INPUTS_A_K: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift scalar or vector.",
    },
];

const CIRCSHIFT_INPUTS_A_K_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift scalar or vector.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension index scalar or dimension vector.",
    },
];

const CIRCSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = circshift(A, K)",
        inputs: &CIRCSHIFT_INPUTS_A_K,
        outputs: &CIRCSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = circshift(A, K, dim)",
        inputs: &CIRCSHIFT_INPUTS_A_K_DIM,
        outputs: &CIRCSHIFT_OUTPUT,
    },
];

pub const CIRCSHIFT_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCSHIFT.TOO_MANY_INPUTS",
    identifier: Some("RunMat:circshift:TooManyInputs"),
    when: "More than three input arguments are supplied.",
    message: "circshift: too many input arguments",
};

pub const CIRCSHIFT_ERROR_INVALID_SHIFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCSHIFT.INVALID_SHIFT",
    identifier: Some("RunMat:circshift:InvalidShift"),
    when: "Shift argument is not a finite real integer scalar/vector.",
    message: "circshift: invalid shift argument",
};

pub const CIRCSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:circshift:InvalidDimensions"),
    when: "Dimension argument is invalid or not compatible with shift argument.",
    message: "circshift: invalid dimension argument",
};

pub const CIRCSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:circshift:UnsupportedInput"),
    when: "Input value type is unsupported for circshift.",
    message: "circshift: unsupported input type",
};

pub const CIRCSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CIRCSHIFT.INTERNAL",
    identifier: Some("RunMat:circshift:Internal"),
    when: "Internal conversion/allocation/provider operations fail.",
    message: "circshift: internal operation failed",
};

const CIRCSHIFT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    CIRCSHIFT_ERROR_TOO_MANY_INPUTS,
    CIRCSHIFT_ERROR_INVALID_SHIFT,
    CIRCSHIFT_ERROR_INVALID_DIMS,
    CIRCSHIFT_ERROR_UNSUPPORTED_INPUT,
    CIRCSHIFT_ERROR_INTERNAL,
];

pub const CIRCSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CIRCSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CIRCSHIFT_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Array elements are permuted without changing or materializing their native integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "K",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Shift scalars/vectors are parsed without binary64 conversion. Native integer magnitudes outside the platform signed shift range are rejected before provider work; floating controls must be finite integer-valued and in range.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dimension scalars/vectors are parsed exactly, must be positive, and cannot contain duplicates.",
    },
];

pub const CIRCSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = circshift(A, K, dim)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All eight integer data classes preserve exact storage and shape. Typed integer controls are decoded exactly; out-of-platform shift magnitudes are rejected. Provider execution or exact gather/re-upload preserves explicit residency where supported.",
    }];
