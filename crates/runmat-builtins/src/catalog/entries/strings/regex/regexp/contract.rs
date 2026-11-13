use crate::*;

const REGEXP_OUTPUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Primary regexp output (depends on options and requested output count).",
}];

const REGEXP_OUTPUT_MULTI: [BuiltinParamDescriptor; 7] = [
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

const REGEXP_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
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

const REGEXP_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const REGEXP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "out = regexp(subject, pattern)",
        inputs: &REGEXP_INPUTS_CORE,
        outputs: &REGEXP_OUTPUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "out = regexp(subject, pattern, options...)",
        inputs: &REGEXP_INPUTS_OPTIONS,
        outputs: &REGEXP_OUTPUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "[start,end_idx,token_extents,match,tokens,names,split] = regexp(subject, pattern, options...)",
        inputs: &REGEXP_INPUTS_OPTIONS,
        outputs: &REGEXP_OUTPUT_MULTI,
    },
];

const REGEXP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXP.INVALID_ARGUMENT",
    identifier: Some("RunMat:regexp:InvalidArgument"),
    when: "Input/options are malformed or unsupported.",
    message: "regexp: invalid argument",
};

const REGEXP_ERROR_PATTERN_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXP.PATTERN_INVALID",
    identifier: Some("RunMat:regexp:PatternInvalid"),
    when: "Pattern cannot be compiled as a regular expression.",
    message: "regexp: invalid regular expression pattern",
};

const REGEXP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXP.INTERNAL",
    identifier: Some("RunMat:regexp:Internal"),
    when: "Internal regexp output assembly fails.",
    message: "regexp: internal operation failed",
};

const REGEXP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    REGEXP_ERROR_INVALID_ARGUMENT,
    REGEXP_ERROR_PATTERN_INVALID,
    REGEXP_ERROR_INTERNAL,
];

pub const REGEXP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGEXP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REGEXP_ERRORS,
};

pub const REGEXP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "regexp accepts host string, character, and cell text for subjects, expressions, output keys, and options. Numeric match indices are outputs only; numeric, logical, symbolic, and provider-resident inputs reject before gather or provider access.",
};
