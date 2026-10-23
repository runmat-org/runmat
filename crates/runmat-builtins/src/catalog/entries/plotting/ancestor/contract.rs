use crate::*;
const ANCESTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching graphics ancestor handle, or an empty handle array when none exists.",
}];

const ANCESTOR_INPUTS_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle or array of graphics handles to inspect.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object type name or cell/string array of type names.",
    },
];

const ANCESTOR_INPUTS_TOPLEVEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle or array of graphics handles to inspect.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object type name or cell/string array of type names.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Use 'toplevel' to return the highest matching ancestor.",
    },
];

const ANCESTOR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = ancestor(h, type)",
        inputs: &ANCESTOR_INPUTS_TYPE,
        outputs: &ANCESTOR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = ancestor(h, type, 'toplevel')",
        inputs: &ANCESTOR_INPUTS_TOPLEVEL,
        outputs: &ANCESTOR_OUTPUT,
    },
];

pub const ANCESTOR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANCESTOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:ancestor:InvalidArgument"),
    when: "Arguments are missing, the type selector is invalid, or the option is not 'toplevel'.",
    message: "ancestor: invalid argument",
};

const ANCESTOR_ERRORS: [BuiltinErrorDescriptor; 2] = [
    ANCESTOR_ERROR_INVALID_ARGUMENT,
    ANCESTOR_ERROR_TOO_MANY_OUTPUTS,
];

pub const ANCESTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANCESTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ANCESTOR_ERRORS,
};

pub const ANCESTOR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "ancestor accepts graphics objects plus text type selectors and returns a graphics object or empty handle array. Integer scalars and arrays are non-graphics inputs and return empty; RunMat's host-double handle encoding is opaque object state, not integer data.",
};

pub const ANCESTOR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANCESTOR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ancestor:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "ancestor: too many output arguments",
};
