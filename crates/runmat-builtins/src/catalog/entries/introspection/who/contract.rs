use crate::*;

const WHO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "names",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of matching variable names.",
}];

const WHO_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const WHO_SIG_SELECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "selector",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name filters, wildcard patterns, or options.",
}];

const WHO_SIG_FILE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-file\""),
        description: "File mode option.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MAT-file path.",
    },
];

const WHO_SIG_REGEXP_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "regexp_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-regexp\""),
        description: "Regexp mode option.",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Regular-expression patterns.",
    },
];

const WHO_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "who(selector, ...) [discarded output]",
        inputs: &WHO_SIG_SELECTOR_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "names = who()",
        inputs: &WHO_SIG_NO_INPUTS,
        outputs: &WHO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "names = who(selector, ...)",
        inputs: &WHO_SIG_SELECTOR_INPUTS,
        outputs: &WHO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "names = who(\"-file\", filename, selector, ...)",
        inputs: &WHO_SIG_FILE_INPUTS,
        outputs: &WHO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "names = who(\"-regexp\", pattern, ...)",
        inputs: &WHO_SIG_REGEXP_INPUTS,
        outputs: &WHO_OUTPUT,
    },
];

const WHO_ERRORS: [BuiltinErrorDescriptor; 12] = [
    BuiltinErrorDescriptor {
        code: "RM.WHO.FILE_REQUIRES_FILENAME",
        identifier: None,
        when: "The '-file' option is provided without a filename.",
        message: "who: '-file' requires a filename",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.FILE_DUPLICATE",
        identifier: None,
        when: "The '-file' option is provided more than once.",
        message: "who: '-file' may only be specified once",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.REGEXP_REQUIRES_PATTERN",
        identifier: None,
        when: "The '-regexp' option is provided without patterns.",
        message: "who: '-regexp' requires at least one pattern",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.REGEXP_EMPTY_PATTERN",
        identifier: None,
        when: "The '-regexp' option receives only empty patterns.",
        message: "who: '-regexp' requires non-empty pattern strings",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.REGEXP_INVALID",
        identifier: None,
        when: "A regexp pattern fails to compile.",
        message: "who: invalid regular expression",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.OPTION_UNSUPPORTED",
        identifier: None,
        when: "An unsupported option token is provided.",
        message: "who: unsupported option",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.FILENAME_ARG_TYPE",
        identifier: None,
        when: "Filename argument is not a char row or string scalar.",
        message: "who: filename must be a character vector or string scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.SELECTOR_CELL_MEMBER_TYPE",
        identifier: None,
        when: "A selector cell contains a non-string-like element.",
        message: "who: selection cells must contain string or character scalars",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.SELECTOR_ARG_TYPE",
        identifier: None,
        when: "Selector argument is not a char array/string/string array/cell thereof.",
        message: "who: selections must be character vectors, string scalars, string arrays, or cell arrays of those types",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.PATTERN_INVALID",
        identifier: None,
        when: "Wildcard selector pattern fails to parse.",
        message: "who: invalid pattern",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.FILE_READ", identifier: None,
        when: "The selected MAT-file cannot be read or decoded.",
        message: "who: MAT-file read failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHO.RESULT_SHAPE", identifier: None,
        when: "The result cell cannot be constructed.",
        message: "who: failed to construct the name cell",
    },
];

pub const WHO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WHO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WHO_ERRORS,
};

pub const WHO_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Integer workspace values are not inspected; an integer selector is invalid text and rejects before provider access. Names are returned as strings without numeric conversion.",
};
