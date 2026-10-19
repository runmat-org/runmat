use crate::*;

const REGEXPI_OUTPUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Primary regexpi output (depends on options and requested output count).",
}];

const REGEXPI_OUTPUT_MULTI: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "1-based match start indices.",
    },
    BuiltinParamDescriptor {
        name: "end_idx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "1-based inclusive match end indices.",
    },
    BuiltinParamDescriptor {
        name: "token_extents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "1-based start and end indices of capture tokens.",
    },
    BuiltinParamDescriptor {
        name: "match",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Matched substrings.",
    },
    BuiltinParamDescriptor {
        name: "tokens",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Capture-group token outputs.",
    },
    BuiltinParamDescriptor {
        name: "names",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Named capture-group outputs.",
    },
    BuiltinParamDescriptor {
        name: "split",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Split output around regex matches.",
    },
];

const REGEXPI_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (char/string/string-array/cellstr).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regular-expression pattern.",
    },
];

const REGEXPI_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (char/string/string-array/cellstr).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Regular-expression pattern.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Output selectors and regexp options.",
    },
];

const REGEXPI_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "out = regexpi(subject, pattern)",
        inputs: &REGEXPI_INPUTS_CORE,
        outputs: &REGEXPI_OUTPUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "out = regexpi(subject, pattern, options...)",
        inputs: &REGEXPI_INPUTS_OPTIONS,
        outputs: &REGEXPI_OUTPUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "[start,end_idx,token_extents,match,tokens,names,split] = regexpi(subject, pattern, options...)",
        inputs: &REGEXPI_INPUTS_OPTIONS,
        outputs: &REGEXPI_OUTPUT_MULTI,
    },
];

const REGEXPI_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPI.INVALID_ARGUMENT",
    identifier: Some("RunMat:regexpi:InvalidArgument"),
    when: "Input/options are malformed or unsupported.",
    message: "regexpi: invalid argument",
};

const REGEXPI_ERROR_PATTERN_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPI.PATTERN_INVALID",
    identifier: Some("RunMat:regexpi:PatternInvalid"),
    when: "Pattern cannot be compiled as a regular expression.",
    message: "regexpi: invalid regular expression pattern",
};

const REGEXPI_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPI.INTERNAL",
    identifier: Some("RunMat:regexpi:Internal"),
    when: "Internal regexpi output assembly fails.",
    message: "regexpi: internal operation failed",
};

const REGEXPI_ERRORS: [BuiltinErrorDescriptor; 3] = [
    REGEXPI_ERROR_INVALID_ARGUMENT,
    REGEXPI_ERROR_PATTERN_INVALID,
    REGEXPI_ERROR_INTERNAL,
];

pub const REGEXPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGEXPI_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REGEXPI_ERRORS,
};

pub const REGEXPI_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "regexpi accepts host string, character, and cell text for subjects, expressions, output keys, and options. Numeric match indices are outputs only; numeric, logical, symbolic, and provider-resident inputs reject through the shared regexp admission boundary before gather or provider access.",
};
