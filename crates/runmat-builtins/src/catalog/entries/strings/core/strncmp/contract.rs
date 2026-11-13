use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical prefix-comparison result.",
}];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First text operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second text operand.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum number of leading characters to compare.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = strncmp(A, B, N)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRNCMP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRNCMP.INVALID_INPUT",
    identifier: Some("RunMat:strncmp:InvalidInput"),
    when: "At least one text input is not a supported text container.",
    message: "strncmp: text inputs must be string/char/cell/string-array values",
};
pub const STRNCMP_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRNCMP.SHAPE_MISMATCH",
    identifier: Some("RunMat:strncmp:ShapeMismatch"),
    when: "Text inputs are not broadcast-compatible.",
    message: "strncmp: input sizes are not broadcast-compatible",
};
pub const STRNCMP_ERROR_INVALID_PREFIX_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRNCMP.INVALID_PREFIX_LENGTH",
    identifier: Some("RunMat:strncmp:InvalidPrefixLength"),
    when: "Prefix length is not a finite integral scalar representable as a host count.",
    message: "strncmp: prefix length must be a finite nonnegative integer scalar",
};
pub const STRNCMP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRNCMP.INTERNAL",
    identifier: Some("RunMat:strncmp:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "strncmp: internal error",
};
pub const STRNCMP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRNCMP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strncmp:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strncmp: too many output arguments",
};
pub const STRNCMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STRNCMP_ERROR_INVALID_INPUT,
        STRNCMP_ERROR_SHAPE_MISMATCH,
        STRNCMP_ERROR_INVALID_PREFIX_LENGTH,
        STRNCMP_ERROR_INTERNAL,
        STRNCMP_ERROR_TOO_MANY_OUTPUTS,
    ],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "N accepts double, single, and every built-in integer class. Negative values are treated as zero; values beyond the host count domain reject without rounding.",
}];
pub const STRNCMP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf = strncmp(A, B, N)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Logical,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "N is parsed directly from authoritative integer storage. Unsupported numeric A or B operands return scalar logical false before provider access or N parsing.",
}];
