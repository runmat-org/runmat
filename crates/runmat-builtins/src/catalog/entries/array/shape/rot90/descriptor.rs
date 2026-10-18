use crate::*;

const ROT90_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array rotated by multiples of 90 degrees around dimensions 1 and 2.",
}];

const ROT90_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix or N-D array.",
}];

const ROT90_INPUTS_A_K: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix or N-D array.",
    },
    BuiltinParamDescriptor {
        name: "k_or_direction",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Rotation count K or direction string ('clockwise'/'counterclockwise').",
    },
];

const ROT90_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = rot90(A)",
        inputs: &ROT90_INPUTS_A,
        outputs: &ROT90_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = rot90(A, k_or_direction)",
        inputs: &ROT90_INPUTS_A_K,
        outputs: &ROT90_OUTPUT,
    },
];

pub const ROT90_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROT90.TOO_MANY_INPUTS",
    identifier: Some("RunMat:rot90:TooManyInputs"),
    when: "More than one optional rotation argument is provided.",
    message: "rot90: too many input arguments",
};

pub const ROT90_ERROR_INVALID_ROTATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROT90.INVALID_ROTATION",
    identifier: Some("RunMat:rot90:InvalidRotation"),
    when: "Rotation count/direction argument is invalid.",
    message: "rot90: invalid rotation argument",
};

pub const ROT90_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROT90.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:rot90:UnsupportedInput"),
    when: "Input value type is unsupported.",
    message: "rot90: unsupported input type",
};

pub const ROT90_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROT90.INTERNAL",
    identifier: Some("RunMat:rot90:Internal"),
    when: "Internal rotation or provider fallback path fails.",
    message: "rot90: internal operation failed",
};

const ROT90_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ROT90_ERROR_TOO_MANY_INPUTS,
    ROT90_ERROR_INVALID_ROTATION,
    ROT90_ERROR_UNSUPPORTED_INPUT,
    ROT90_ERROR_INTERNAL,
];

pub const ROT90_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROT90_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ROT90_ERRORS,
};

pub const ROT90_DIRECTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rot90-direction-token",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rot90 accepts textual direction tokens as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Rot90DirectionTokenExtension"),
};
pub const ROT90_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ROT90_DIRECTION_EXTENSION];
const ROT90_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents all eight integer array classes; rotation is a pure permutation of authoritative native storage.",
    },
    BuiltinIntegerInputCapability {
        name: "K",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The integer rotation count is reduced modulo four directly in its native signed or unsigned domain.",
    },
];
pub const ROT90_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = rot90(integer_A [, integer_K])",
        inputs: &ROT90_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "No numeric conversion occurs. Host storage is reordered exactly; resident integer fallback gathers and re-uploads through the exact owning provider when direct permutation hooks are unavailable.",
    }];
