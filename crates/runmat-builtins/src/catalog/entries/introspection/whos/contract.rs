use crate::*;

const WHOS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "vars",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure array describing matching variables.",
}];

const WHOS_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const WHOS_SIG_SELECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "selector",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name filters, wildcard patterns, or options.",
}];

const WHOS_SIG_FILE_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const WHOS_SIG_REGEXP_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const WHOS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "whos(selector, ...) [discarded output]",
        inputs: &WHOS_SIG_SELECTOR_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "vars = whos()",
        inputs: &WHOS_SIG_NO_INPUTS,
        outputs: &WHOS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "vars = whos(selector, ...)",
        inputs: &WHOS_SIG_SELECTOR_INPUTS,
        outputs: &WHOS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "vars = whos(\"-file\", filename, selector, ...)",
        inputs: &WHOS_SIG_FILE_INPUTS,
        outputs: &WHOS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "vars = whos(\"-regexp\", pattern, ...)",
        inputs: &WHOS_SIG_REGEXP_INPUTS,
        outputs: &WHOS_OUTPUT,
    },
];

const WHOS_ERRORS: [BuiltinErrorDescriptor; 15] = [
    BuiltinErrorDescriptor {
        code: "RM.WHOS.FILE_REQUIRES_FILENAME",
        identifier: None,
        when: "The '-file' option is provided without a filename.",
        message: "whos: '-file' requires a filename",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.FILE_DUPLICATE",
        identifier: None,
        when: "The '-file' option is provided more than once.",
        message: "whos: '-file' may only be specified once",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.REGEXP_REQUIRES_PATTERN",
        identifier: None,
        when: "The '-regexp' option is provided without patterns.",
        message: "whos: '-regexp' requires at least one pattern",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.REGEXP_EMPTY_PATTERN",
        identifier: None,
        when: "The '-regexp' option receives only empty patterns.",
        message: "whos: '-regexp' requires non-empty pattern strings",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.REGEXP_INVALID",
        identifier: None,
        when: "A regexp pattern fails to compile.",
        message: "whos: invalid regular expression",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.OPTION_UNSUPPORTED",
        identifier: None,
        when: "An unsupported option token is provided.",
        message: "whos: unsupported option",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.FILENAME_ARG_TYPE",
        identifier: None,
        when: "Filename argument is not a char row or string scalar.",
        message: "whos: filename must be a character vector or string scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.SELECTOR_CELL_MEMBER_TYPE",
        identifier: None,
        when: "A selector cell contains a non-string-like element.",
        message: "whos: selection cells must contain string or character scalars",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.SELECTOR_ARG_TYPE",
        identifier: None,
        when: "Selector argument is not a char array/string/string array/cell thereof.",
        message: "whos: selections must be character vectors, string scalars, string arrays, or cell arrays of those types",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.PATTERN_INVALID",
        identifier: None,
        when: "Wildcard selector pattern fails to parse.",
        message: "whos: invalid pattern",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.FILE_READ", identifier: None,
        when: "The selected MAT-file cannot be read or decoded.",
        message: "whos: MAT-file read failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.SIZE", identifier: None,
        when: "A visible value's shape cannot be inspected or represented as a size row.",
        message: "whos: failed to materialize size vector",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.INVALID_HANDLE", identifier: None,
        when: "A handle target is unavailable during recursive byte accounting.",
        message: "whos: invalid handle object target",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.INVALID_LISTENER", identifier: None,
        when: "A listener target or callback is unavailable during recursive byte accounting.",
        message: "whos: invalid listener target or callback",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHOS.RESULT_SHAPE", identifier: None,
        when: "A metadata record or result struct array cannot be constructed.",
        message: "whos: failed to construct metadata records",
    },
];

pub const WHOS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WHOS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WHOS_ERRORS,
};

pub const WHOS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Integer selectors are invalid text. Integer values already in the workspace or MAT-file keep their exact class and width in metadata; `bytes` and `size` are presentation doubles and may round values beyond binary64's exact-integer range.",
};
