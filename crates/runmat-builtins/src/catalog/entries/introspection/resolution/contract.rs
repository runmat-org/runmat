use crate::*;

/// Lexical option token shared by runtime argument admission and catalog
/// inference. `None` means a name spelling; unknown dash-prefixed text still
/// occupies an option position and is rejected by runtime option decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichOptionToken {
    All,
    Builtin,
    Variable,
    File,
    Unrecognized,
}

pub fn which_option_token(text: &str) -> Option<WhichOptionToken> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "-all" => Some(WhichOptionToken::All),
        "-builtin" | "-built-in" => Some(WhichOptionToken::Builtin),
        "-var" | "-variable" => Some(WhichOptionToken::Variable),
        "-file" => Some(WhichOptionToken::File),
        _ if text.trim_start().starts_with('-') => Some(WhichOptionToken::Unrecognized),
        _ => None,
    }
}
pub const WHICH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This identity accepts structural text controls or no inputs; numeric integer values are not converted to names or version controls.",
};
const WHICH_OUTPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single resolution result string.",
}];

const WHICH_OUTPUT_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "All matching resolution strings (cell array of char rows).",
}];

const WHICH_SIG_NAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Name to resolve.",
}];

const WHICH_SIG_NAME_OPTION_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name to resolve.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional mode filter (-all|-builtin|-var|-file).",
    },
];

const WHICH_SIG_OPTION_NAME_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mode filter (-all|-builtin|-var|-file).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name to resolve.",
    },
];

const WHICH_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "result = which(name)",
        inputs: &WHICH_SIG_NAME_INPUTS,
        outputs: &WHICH_OUTPUT_SINGLE,
    },
    BuiltinSignatureDescriptor {
        label: "result = which(name, option)",
        inputs: &WHICH_SIG_NAME_OPTION_INPUTS,
        outputs: &WHICH_OUTPUT_SINGLE,
    },
    BuiltinSignatureDescriptor {
        label: "result = which(option, name)",
        inputs: &WHICH_SIG_OPTION_NAME_INPUTS,
        outputs: &WHICH_OUTPUT_SINGLE,
    },
    BuiltinSignatureDescriptor {
        label: "results = which(\"-all\", name)",
        inputs: &WHICH_SIG_OPTION_NAME_INPUTS,
        outputs: &WHICH_OUTPUT_ALL,
    },
];

pub const WHICH_ERROR_ARG_COUNT_MIN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.ARG_COUNT_MIN",
    identifier: None,
    when: "No input arguments are provided.",
    message: "which: not enough input arguments",
};

pub const WHICH_ERROR_ARG_COUNT_MAX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.ARG_COUNT_MAX",
    identifier: None,
    when: "More than one name argument is provided.",
    message: "which: too many input arguments",
};

pub const WHICH_ERROR_NAME_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.NAME_ARG_TYPE",
    identifier: None,
    when: "Name argument is not a char row or string scalar.",
    message: "which: name must be a character vector or string scalar",
};

pub const WHICH_ERROR_OPTION_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.OPTION_ARG_TYPE",
    identifier: None,
    when: "Option argument is not a char row or string scalar.",
    message: "which: option must be a character vector or string scalar",
};

pub const WHICH_ERROR_OPTION_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.OPTION_CONFLICT",
    identifier: None,
    when: "Conflicting mode options are combined.",
    message: "which: conflicting option",
};

pub const WHICH_ERROR_OPTION_UNRECOGNIZED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHICH.OPTION_UNRECOGNIZED",
    identifier: None,
    when: "An unrecognized mode option is provided.",
    message: "which: unrecognized option",
};

const WHICH_ERRORS: [BuiltinErrorDescriptor; 9] = [
    WHICH_ERROR_ARG_COUNT_MIN,
    WHICH_ERROR_ARG_COUNT_MAX,
    WHICH_ERROR_NAME_ARG_TYPE,
    WHICH_ERROR_OPTION_ARG_TYPE,
    WHICH_ERROR_OPTION_CONFLICT,
    WHICH_ERROR_OPTION_UNRECOGNIZED,
    BuiltinErrorDescriptor {
        code: "RM.WHICH.WORKSPACE",
        identifier: None,
        when: "Workspace lookup authority rejects the query.",
        message: "which: workspace lookup failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHICH.FILESYSTEM",
        identifier: None,
        when: "Search-path or filesystem resolution fails.",
        message: "which: path lookup failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.WHICH.RESULT",
        identifier: None,
        when: "The result cell or display text cannot be constructed.",
        message: "which: result construction failed",
    },
];

pub const WHICH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WHICH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WHICH_ERRORS,
};
