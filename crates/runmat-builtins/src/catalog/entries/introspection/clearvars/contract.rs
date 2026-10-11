use crate::*;
pub const CLEARVARS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This identity accepts structural text controls or no inputs; numeric integer values are not converted to names or version controls.",
};
const CLEARVARS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty 0x0 return value.",
}];

const CLEARVARS_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CLEARVARS_SIG_WORD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name_or_option",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable names and options such as '-except'.",
}];

const CLEARVARS_NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];
const CLEARVARS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "clearvars",
        inputs: &CLEARVARS_SIG_NO_INPUTS,
        outputs: &CLEARVARS_NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "clearvars(name_or_option, ...)",
        inputs: &CLEARVARS_SIG_WORD_INPUTS,
        outputs: &CLEARVARS_NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "ans = clearvars()",
        inputs: &CLEARVARS_SIG_NO_INPUTS,
        outputs: &CLEARVARS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ans = clearvars(name_or_option, ...)",
        inputs: &CLEARVARS_SIG_WORD_INPUTS,
        outputs: &CLEARVARS_OUTPUT,
    },
];

const CLEARVARS_ERRORS: [BuiltinErrorDescriptor; 8] = [
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.DUP_EXCEPT",
        identifier: None,
        when: "The '-except' option appears more than once.",
        message: "clearvars: duplicate -except option",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.UNSUPPORTED_OPTION",
        identifier: None,
        when: "An unknown option beginning with '-' is provided.",
        message: "clearvars: unsupported option",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.EXCEPT_REQUIRES_NAMES",
        identifier: None,
        when: "The '-except' option is provided without exclusion names.",
        message: "clearvars: -except requires at least one variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.WORKSPACE_UNAVAILABLE",
        identifier: None,
        when: "Workspace snapshot is unavailable while processing '-except'.",
        message: "clearvars: workspace state unavailable",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.CHAR_ROWS",
        identifier: None,
        when: "Character-array argument is not a row vector.",
        message: "clearvars: character array inputs must be a row vector or scalar text value",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.NAME_ARG_TYPE",
        identifier: None,
        when: "An argument is not a string scalar/array or row character vector.",
        message: "clearvars: expected variable names as character vectors or string scalars",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.WORKSPACE",
        identifier: None,
        when: "The current workspace cannot be read or mutated.",
        message: "clearvars: workspace operation failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.CLEANUP",
        identifier: None,
        when: "An onCleanup callback fails while removing a binding.",
        message: "clearvars: cleanup callback failed",
    },
];

pub const CLEARVARS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEARVARS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEARVARS_ERRORS,
};
