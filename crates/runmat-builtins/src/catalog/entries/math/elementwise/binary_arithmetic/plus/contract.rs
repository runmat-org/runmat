use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, LIKE, OUTPUT, PROTOTYPE};
use crate::*;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const INFERENCE_POLICY:
    super::super::inference::BinaryArithmeticInferencePolicy =
    super::super::inference::MATCHING_SPARSE_REAL_RESULT;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = plus(A, B)",
        inputs: &[INPUT_A, INPUT_B],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "C = plus(A, B, \"like\", prototype)",
        inputs: &[INPUT_A, INPUT_B, LIKE, PROTOTYPE],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "S = plus(A, B)",
        inputs: &[
            BuiltinParamDescriptor { name: "A", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Host string or convertible numeric, logical, character, or scalar-convertible cell data; at least one operand must be a string." },
            BuiltinParamDescriptor { name: "B", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Second host operand; converted string shapes must permit implicit expansion." },
        ],
        outputs: &[BuiltinParamDescriptor { name: "S", ty: BuiltinParamType::StringScalar, arity: BuiltinParamArity::Required, default: None, description: "String scalar or array with the broadcasted converted shape; missing values propagate." }],
    },
];

pub const PLUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:plus:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "plus: invalid argument",
};
pub const PLUS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INVALID_INPUT",
    identifier: Some("RunMat:plus:InvalidInput"),
    when: "Operands or prototypes cannot be converted into the selected numeric, symbolic, or host string form.",
    message: "plus: invalid input",
};
pub const PLUS_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SIZE_MISMATCH",
    identifier: Some("RunMat:plus:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "plus: array sizes are not compatible for broadcasting",
};
pub const PLUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INTERNAL",
    identifier: Some("RunMat:plus:Internal"),
    when: "Provider interaction, gather, upload, or tensor construction fails.",
    message: "plus: internal error",
};
pub const PLUS_ERROR_SPARSE_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SPARSE_SIZE_MISMATCH",
    identifier: Some("RunMat:plus:SparseSizeMismatch"),
    when: "Sparse operands cannot expand to a compatible result shape.",
    message: "plus: sparse operand sizes are not compatible",
};
pub const PLUS_ERROR_SPARSE_UNSUPPORTED_OPERAND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SPARSE_UNSUPPORTED_OPERAND",
    identifier: Some("RunMat:plus:SparseUnsupportedOperand"),
    when: "Sparse arithmetic receives an unsupported operand class or residency.",
    message: "plus: unsupported sparse arithmetic operand",
};
pub const PLUS_ERROR_SPARSE_DENSIFY_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SPARSE_DENSIFY_TOO_LARGE",
    identifier: Some("RunMat:plus:SparseDensifyTooLarge"),
    when: "The result would exceed the sparse or dense materialization limit.",
    message: "plus: sparse arithmetic result is too large to materialize",
};
pub const PLUS_ERROR_SPARSE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SPARSE_INTERNAL",
    identifier: Some("RunMat:plus:SparseInternal"),
    when: "Sparse storage construction or conversion fails.",
    message: "plus: sparse arithmetic internal error",
};

pub const PLUS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plus:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "plus: too many output arguments",
};

pub const PLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        PLUS_ERROR_TOO_MANY_OUTPUTS,
        PLUS_ERROR_INVALID_ARGUMENT,
        PLUS_ERROR_INVALID_INPUT,
        PLUS_ERROR_SIZE_MISMATCH,
        PLUS_ERROR_INTERNAL,
        PLUS_ERROR_SPARSE_SIZE_MISMATCH,
        PLUS_ERROR_SPARSE_UNSUPPORTED_OPERAND,
        PLUS_ERROR_SPARSE_DENSIFY_TOO_LARGE,
        PLUS_ERROR_SPARSE_INTERNAL,
    ],
};

pub const PLUS_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "plus-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "plus with a 'like' output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PlusLikePrototypeExtension"),
};
pub const PLUS_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[PLUS_LIKE_EXTENSION];

pub const PLUS_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[
        BuiltinIntegerCapabilityDescriptor { form: "C = plus(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "Same-class integer sums are exact and saturating. Scalar-double arithmetic uses MATLAB rounding, including the extended-precision 64-bit rule; resident paths use native kernels when supported and otherwise gather authoritative storage." },
        BuiltinIntegerCapabilityDescriptor { form: "S = plus(A, B)", inputs: &[
            BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "An integer operand converts directly from its native class when the other operand is a string; all integer bits contribute to decimal text." },
            BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "String addition preserves exact integer text and applies implicit expansion to the converted shapes." },
        ], computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "String addition returns host strings, not integer sums. Explicitly gather device values first; numeric output prototypes do not apply." },
    ];
