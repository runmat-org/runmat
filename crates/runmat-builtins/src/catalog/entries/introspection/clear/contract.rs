use crate::*;
pub const CLEAR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This identity accepts structural text controls or no inputs; numeric integer values are not converted to names or version controls.",
};
const CLEAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty 0x0 return value.",
}];

const CLEAR_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CLEAR_SIG_NAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable names to clear. 'all' clears entire workspace.",
}];

const CLEAR_NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];
const CLEAR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "clear",
        inputs: &CLEAR_SIG_NO_INPUTS,
        outputs: &CLEAR_NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "clear(name, ...)",
        inputs: &CLEAR_SIG_NAME_INPUTS,
        outputs: &CLEAR_NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "ans = clear()",
        inputs: &CLEAR_SIG_NO_INPUTS,
        outputs: &CLEAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ans = clear(name, ...)",
        inputs: &CLEAR_SIG_NAME_INPUTS,
        outputs: &CLEAR_OUTPUT,
    },
];

const CLEAR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.CHAR_ROWS",
        identifier: None,
        when: "Character-array argument is not a row vector.",
        message: "clear: character array inputs must be a row vector or scalar text value",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.NAME_ARG_TYPE",
        identifier: None,
        when: "A target argument is not a string scalar/array or row character vector.",
        message: "clear: expected variable names as character vectors or string scalars",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.WORKSPACE",
        identifier: None,
        when: "The current workspace cannot be read or mutated.",
        message: "clear: workspace operation failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.CLEANUP",
        identifier: None,
        when: "An onCleanup callback or dynamic function invalidation fails.",
        message: "clear: cleanup or function-cache invalidation failed",
    },
];

pub const CLEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEAR_ERRORS,
};
