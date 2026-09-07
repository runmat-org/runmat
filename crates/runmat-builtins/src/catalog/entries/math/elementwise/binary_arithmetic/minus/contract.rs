use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const INFERENCE_POLICY:
    super::super::inference::BinaryArithmeticInferencePolicy = super::super::inference::REAL_RESULT;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = minus(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = minus(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
];

pub const MINUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:minus:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "minus: invalid argument",
};
pub const MINUS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INVALID_INPUT",
    identifier: Some("RunMat:minus:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric or symbolic forms.",
    message: "minus: invalid input",
};
pub const MINUS_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SIZE_MISMATCH",
    identifier: Some("RunMat:minus:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "minus: array sizes are not compatible for broadcasting",
};
pub const MINUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INTERNAL",
    identifier: Some("RunMat:minus:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "minus: internal error",
};
pub const MINUS_ERROR_SPARSE_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_SIZE_MISMATCH",
    identifier: Some("RunMat:minus:SparseSizeMismatch"),
    when: "Sparse operands cannot expand to a compatible result shape.",
    message: "minus: sparse operand sizes are not compatible",
};
pub const MINUS_ERROR_SPARSE_UNSUPPORTED_OPERAND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_UNSUPPORTED_OPERAND",
    identifier: Some("RunMat:minus:SparseUnsupportedOperand"),
    when: "Sparse arithmetic receives an unsupported operand class or residency.",
    message: "minus: unsupported sparse arithmetic operand",
};
pub const MINUS_ERROR_SPARSE_DENSIFY_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_DENSIFY_TOO_LARGE",
    identifier: Some("RunMat:minus:SparseDensifyTooLarge"),
    when: "The result would exceed the sparse or dense materialization limit.",
    message: "minus: sparse arithmetic result is too large to materialize",
};
pub const MINUS_ERROR_SPARSE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_INTERNAL",
    identifier: Some("RunMat:minus:SparseInternal"),
    when: "Sparse storage construction or conversion fails.",
    message: "minus: sparse arithmetic internal error",
};

pub const MINUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MINUS_ERROR_INVALID_ARGUMENT,
        MINUS_ERROR_INVALID_INPUT,
        MINUS_ERROR_SIZE_MISMATCH,
        MINUS_ERROR_INTERNAL,
        MINUS_ERROR_SPARSE_SIZE_MISMATCH,
        MINUS_ERROR_SPARSE_UNSUPPORTED_OPERAND,
        MINUS_ERROR_SPARSE_DENSIFY_TOO_LARGE,
        MINUS_ERROR_SPARSE_INTERNAL,
    ],
};

pub const MINUS_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "minus-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "minus with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MinusLikePrototypeExtension"),
};
pub const MINUS_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[MINUS_LIKE_EXTENSION];
pub const MINUS_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = minus(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Same-class integer differences are exact and saturating. Scalar-double arithmetic uses MATLAB rounding, including the extended-precision 64-bit rule; resident paths use native kernels when supported and otherwise gather authoritative storage." }];
