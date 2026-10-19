use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varname",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Valid variable name text with the same container form as the input.",
}];

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text, string array, character array, or cell array of text scalars.",
}];

const INPUTS_EXCLUSIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text, string array, character array, or cell array of text scalars.",
    },
    BuiltinParamDescriptor {
        name: "exclusions",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Names that generated output must not match.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "varname = genvarname(str)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "varname = genvarname(str, exclusions)",
        inputs: &INPUTS_EXCLUSIONS,
        outputs: &OUTPUT,
    },
];

pub const GENVARNAME_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INVALID_INPUT",
    identifier: Some("RunMat:genvarname:InvalidInput"),
    when: "Input is not text-compatible or cell elements are not text scalars.",
    message: "genvarname: input must be text or a cell array of text",
};

pub const GENVARNAME_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INVALID_OPTION",
    identifier: Some("RunMat:genvarname:InvalidOption"),
    when: "More than one exclusions argument is supplied.",
    message: "genvarname: expected str or str,exclusions",
};

pub const GENVARNAME_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INTERNAL",
    identifier: Some("RunMat:genvarname:Internal"),
    when: "Internal string, cell, or character array construction failed.",
    message: "genvarname: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GENVARNAME_ERROR_INVALID_INPUT,
    GENVARNAME_ERROR_INVALID_OPTION,
    GENVARNAME_ERROR_INTERNAL,
    GENVARNAME_ERROR_TOO_MANY_OUTPUTS,
];

pub const GENVARNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GENVARNAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "genvarname accepts host text names and exclusions only. All eight integer classes and provider-resident numeric values reject without implicit text conversion, gather, or provider access.",
    };

pub const GENVARNAME_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:genvarname:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "genvarname: too many output arguments",
};
