use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const INFERENCE_POLICY:
    super::super::inference::BinaryArithmeticInferencePolicy =
    super::super::inference::RUNTIME_DEPENDENT_RESULT;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = power(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = power(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
];

pub const POWER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POWER.INVALID_ARGUMENT",
    identifier: Some("RunMat:power:InvalidArgument"),
    when: "Optional arguments or integer exponents are malformed or unsupported.",
    message: "power: invalid argument",
};
pub const POWER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POWER.INVALID_INPUT",
    identifier: Some("RunMat:power:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric or symbolic forms.",
    message: "power: invalid input",
};
pub const POWER_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POWER.SIZE_MISMATCH",
    identifier: Some("RunMat:power:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "power: array sizes are not compatible for broadcasting",
};
pub const POWER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POWER.INTERNAL",
    identifier: Some("RunMat:power:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "power: internal error",
};

pub const POWER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        POWER_ERROR_INVALID_ARGUMENT,
        POWER_ERROR_INVALID_INPUT,
        POWER_ERROR_SIZE_MISMATCH,
        POWER_ERROR_INTERNAL,
    ],
};
pub const POWER_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "power-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "power with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PowerLikePrototypeExtension"),
};
pub const POWER_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[POWER_LIKE_EXTENSION];
pub const POWER_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = power(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Integer bases preserve their class, require nonnegative integer-valued exponents, and use exact saturating exponentiation." }];
