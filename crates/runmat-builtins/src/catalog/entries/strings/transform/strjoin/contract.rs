use crate::*;
const STRJOIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Joined string scalar or character row; the row-wise extension returns a string array.",
}];

const STRJOIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text array.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("' '"),
        description:
            "Text delimiter, or a vector with one delimiter per gap. Omitted delimiter is a space.",
    },
];

const STRJOIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strjoin(text, delimiter)",
    inputs: &STRJOIN_INPUTS,
    outputs: &STRJOIN_OUTPUT,
}];

pub const STRJOIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRJOIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STRJOIN_ERROR_TOO_MANY_OUTPUTS,
        STRJOIN_ERROR_INVALID_INPUT,
        STRJOIN_ERROR_INVALID_DELIMITER,
        STRJOIN_ERROR_ARGUMENT_COUNT,
    ],
};

pub const STRJOIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "out = strjoin(text, integer_scalar_delimiter)",
    inputs: &[BuiltinIntegerInputCapability {
        name: "delimiter", classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The legacy row-wise extension formats native integer scalars exactly as decimal text. Integer tensors, numeric text input, strict mode and provider values remain unsupported.",
    }],
    computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Decimal formatting reads the authoritative integer scalar without floating-point conversion; the output is a host string array.",
}];

pub const STRJOIN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRJOIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strjoin:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strjoin: too many output arguments",
};

pub const STRJOIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRJOIN.INVALID_INPUT",
    identifier: Some("RunMat:strjoin:InvalidInput"),
    when: "Input is not a supported text vector, or has unsupported rank.",
    message: "strjoin: expected a string vector or a cell vector of character rows",
};
pub const STRJOIN_ERROR_INVALID_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRJOIN.INVALID_DELIMITER",
    identifier: Some("RunMat:strjoin:InvalidDelimiter"),
    when: "Delimiter is not text, or its vector length does not match the gaps.",
    message: "strjoin: delimiter must be scalar text or one text element per gap",
};
pub const STRJOIN_ERROR_ARGUMENT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRJOIN.ARGUMENT_COUNT",
    identifier: Some("RunMat:strjoin:ArgumentCount"),
    when: "More than two inputs are supplied.",
    message: "strjoin: expected one or two inputs",
};
pub const STRJOIN_ROWWISE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "strjoin-rowwise", mode: BuiltinExtensionMode::RunMatOnly,
    description: "Explicit scalar delimiters preserve RunMat row-wise joining of string and character arrays.",
    error_identifier: Some("RunMat:compatibility:StrjoinRowwiseExtension"),
};
