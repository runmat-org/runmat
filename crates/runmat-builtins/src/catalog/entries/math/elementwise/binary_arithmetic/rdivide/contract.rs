use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const INFERENCE_POLICY:
    super::super::inference::BinaryArithmeticInferencePolicy = super::super::inference::REAL_RESULT;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = rdivide(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = rdivide(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
];

pub const RDIVIDE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RDIVIDE.INVALID_ARGUMENT",
    identifier: Some("RunMat:rdivide:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "rdivide: invalid argument",
};
pub const RDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:rdivide:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric or symbolic forms.",
    message: "rdivide: invalid input",
};
pub const RDIVIDE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RDIVIDE.SIZE_MISMATCH",
    identifier: Some("RunMat:rdivide:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "rdivide: array sizes are not compatible for broadcasting",
};
pub const RDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RDIVIDE.INTERNAL",
    identifier: Some("RunMat:rdivide:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "rdivide: internal error",
};

pub const RDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        RDIVIDE_ERROR_INVALID_ARGUMENT,
        RDIVIDE_ERROR_INVALID_INPUT,
        RDIVIDE_ERROR_SIZE_MISMATCH,
        RDIVIDE_ERROR_INTERNAL,
    ],
};
pub const RDIVIDE_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rdivide-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rdivide with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RdivideLikePrototypeExtension"),
};
pub const RDIVIDE_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[RDIVIDE_LIKE_EXTENSION];
pub const RDIVIDE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = rdivide(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Integer quotients preserve the integer class, round to nearest with half ties away from zero, and saturate. Resident fallback retains authoritative typed storage." }];
