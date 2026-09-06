use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = times(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = times(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
];

pub const TIMES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INVALID_ARGUMENT",
    identifier: Some("RunMat:times:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "times: invalid argument",
};
pub const TIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INVALID_INPUT",
    identifier: Some("RunMat:times:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric or symbolic forms.",
    message: "times: invalid input",
};
pub const TIMES_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SIZE_MISMATCH",
    identifier: Some("RunMat:times:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "times: array sizes are not compatible for broadcasting",
};
pub const TIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INTERNAL",
    identifier: Some("RunMat:times:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "times: internal error",
};
pub const TIMES_ERROR_SPARSE_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_SIZE_MISMATCH",
    identifier: Some("RunMat:times:SparseSizeMismatch"),
    when: "Sparse operands cannot expand to a compatible result shape.",
    message: "times: sparse operand sizes are not compatible",
};
pub const TIMES_ERROR_SPARSE_UNSUPPORTED_OPERAND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_UNSUPPORTED_OPERAND",
    identifier: Some("RunMat:times:SparseUnsupportedOperand"),
    when: "Sparse arithmetic receives an unsupported operand class or residency.",
    message: "times: unsupported sparse arithmetic operand",
};
pub const TIMES_ERROR_SPARSE_DENSIFY_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_DENSIFY_TOO_LARGE",
    identifier: Some("RunMat:times:SparseDensifyTooLarge"),
    when: "The result would exceed the sparse or dense materialization limit.",
    message: "times: sparse arithmetic result is too large to materialize",
};
pub const TIMES_ERROR_SPARSE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_INTERNAL",
    identifier: Some("RunMat:times:SparseInternal"),
    when: "Sparse storage construction or conversion fails.",
    message: "times: sparse arithmetic internal error",
};

pub const TIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        TIMES_ERROR_INVALID_ARGUMENT,
        TIMES_ERROR_INVALID_INPUT,
        TIMES_ERROR_SIZE_MISMATCH,
        TIMES_ERROR_INTERNAL,
        TIMES_ERROR_SPARSE_SIZE_MISMATCH,
        TIMES_ERROR_SPARSE_UNSUPPORTED_OPERAND,
        TIMES_ERROR_SPARSE_DENSIFY_TOO_LARGE,
        TIMES_ERROR_SPARSE_INTERNAL,
    ],
};
pub const TIMES_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "times-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "times with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TimesLikePrototypeExtension"),
};
pub const TIMES_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[TIMES_LIKE_EXTENSION];
pub const TIMES_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = times(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Same-class integer products are exact and saturating. Scalar-double arithmetic uses MATLAB rounding, including the extended-precision 64-bit rule; resident paths use native kernels when supported and otherwise gather authoritative storage." }];
