use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const INFERENCE_POLICY:
    super::super::inference::BinaryArithmeticInferencePolicy = super::super::inference::REAL_RESULT;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = ldivide(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = ldivide(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
];

pub const LDIVIDE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ldivide:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "ldivide: invalid argument",
};
pub const LDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:ldivide:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric or symbolic forms.",
    message: "ldivide: invalid input",
};
pub const LDIVIDE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.SIZE_MISMATCH",
    identifier: Some("RunMat:ldivide:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "ldivide: array sizes are not compatible for broadcasting",
};
pub const LDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LDIVIDE.INTERNAL",
    identifier: Some("RunMat:ldivide:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "ldivide: internal error",
};

pub const LDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        LDIVIDE_ERROR_INVALID_ARGUMENT,
        LDIVIDE_ERROR_INVALID_INPUT,
        LDIVIDE_ERROR_SIZE_MISMATCH,
        LDIVIDE_ERROR_INTERNAL,
    ],
};
pub const LDIVIDE_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ldivide-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ldivide with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LdivideLikePrototypeExtension"),
};
pub const LDIVIDE_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[LDIVIDE_LIKE_EXTENSION];
pub const LDIVIDE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = ldivide(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "The result is B divided by A. Integer quotients preserve the integer class, round to nearest with half ties away from zero, and saturate." }];
