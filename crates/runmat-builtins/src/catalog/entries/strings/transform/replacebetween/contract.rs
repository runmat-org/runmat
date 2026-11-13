use crate::*;
pub const REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "replacebetween-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "replaceBetween host fallback for explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReplaceBetweenExplicitGpuInputExtension"),
    };
pub const REPLACE_BETWEEN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REPLACE_BETWEEN_EXPLICIT_GPU_EXTENSION];
pub const REPLACE_BETWEEN_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "startPos/endPos",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric positions accept all eight integer classes and are read exactly as one-based inclusive boundaries.",
    }];
pub const REPLACE_BETWEEN_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "str/newText",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Source and replacement text reject integer and resident numeric values before gather or implicit formatting.",
    }];
pub const REPLACE_BETWEEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = replaceBetween(str, integer_startPos, integer_endPos, newText)",
        inputs: &REPLACE_BETWEEN_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions remain exact through range and ordering validation; output preserves the host text container. Automatic residency gathers transparently, while explicit gpuArray fallback is separately gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = replaceBetween(integer_str, start, end, integer_newText)",
        inputs: &REPLACE_BETWEEN_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer data is outside the public text roles and is never converted to decimal text.",
    },
];
pub const REPLACE_BETWEEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = replaceBetween(text, start, end, newText)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "text",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Input host text.",
            },
            BuiltinParamDescriptor {
                name: "start",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "One-based inclusive position or boundary pattern.",
            },
            BuiltinParamDescriptor {
                name: "end",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "One-based inclusive position or boundary pattern.",
            },
            BuiltinParamDescriptor {
                name: "newText",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Replacement text.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "out",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Host result.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        REPLACE_BETWEEN_ERROR_TOO_MANY_OUTPUTS,
        REPLACE_BETWEEN_ERROR_INVALID_INPUT,
        REPLACE_BETWEEN_ERROR_PROCESSING,
        REPLACE_BETWEEN_ERROR_ARGUMENT_COUNT,
    ],
};

pub const REPLACE_BETWEEN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE_BETWEEN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:replaceBetween:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "replaceBetween: too many output arguments",
};

pub const REPLACE_BETWEEN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE_BETWEEN.INVALID_INPUT",
    identifier: Some("RunMat:replaceBetween:InvalidInput"),
    when: "A required text role contains unsupported input data.",
    message: "replaceBetween: expected supported host text input",
};
pub const REPLACE_BETWEEN_ERROR_PROCESSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
 code: "RM.REPLACE_BETWEEN.PROCESSING", identifier: None,
 when: "Shared text/pattern processing rejects unsupported nested input, pattern syntax, geometry, or output construction.",
 message: "replaceBetween: text processing failed",
};

pub const REPLACE_BETWEEN_ERROR_ARGUMENT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE_BETWEEN.ARGUMENT_COUNT",
    identifier: Some("RunMat:replaceBetween:ArgumentCount"),
    when: "The call does not supply exactly text, start, end, and replacement.",
    message: "replaceBetween: expected exactly start, end, and replacement text",
};
