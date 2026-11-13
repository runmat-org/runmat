use crate::*;
pub const ISKEYWORD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This identity accepts structural text controls or no inputs; numeric integer values are not converted to names or version controls.",
};

pub const NAMELENGTHMAX_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This identity accepts structural text controls or no inputs; numeric integer values are not converted to names or version controls.",
};
const OUT_NUMERIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "l",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Maximum supported MATLAB identifier length.",
}];

const OUT_BOOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the text is a MATLAB keyword.",
}];

const OUT_KEYWORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell column of MATLAB keyword character vectors.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar or character vector to test.",
}];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const NAMELENGTHMAX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "l = namelengthmax",
    inputs: &NO_INPUTS,
    outputs: &OUT_NUMERIC,
}];

const ISKEYWORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = iskeyword(s)",
        inputs: &IN_TEXT,
        outputs: &OUT_BOOL,
    },
    BuiltinSignatureDescriptor {
        label: "k = iskeyword",
        inputs: &NO_INPUTS,
        outputs: &OUT_KEYWORDS,
    },
];

pub const ERROR_NAMELENGTHMAX_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAMELENGTHMAX.ARG_COUNT",
    identifier: Some("RunMat:namelengthmax:InvalidArgument"),
    when: "Any input argument is supplied.",
    message: "namelengthmax: expected no input arguments",
};

pub const ERROR_ISKEYWORD_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.ARG_COUNT",
    identifier: Some("RunMat:iskeyword:InvalidArgument"),
    when: "More than one input argument is supplied.",
    message: "iskeyword: expected zero or one input argument",
};

pub const ERROR_ISKEYWORD_TEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.TEXT",
    identifier: Some("RunMat:iskeyword:InvalidText"),
    when: "Input is not a string scalar or character vector.",
    message: "iskeyword: input must be a string scalar or character vector",
};

pub const ERROR_ISKEYWORD_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.INTERNAL",
    identifier: Some("RunMat:iskeyword:Internal"),
    when: "RunMat cannot construct the keyword-list cell array.",
    message: "iskeyword: failed to build keyword list",
};

pub const NAMELENGTHMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAMELENGTHMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ERROR_NAMELENGTHMAX_ARG_COUNT],
};

pub const ISKEYWORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISKEYWORD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ERROR_ISKEYWORD_ARG_COUNT,
        ERROR_ISKEYWORD_TEXT,
        ERROR_ISKEYWORD_INTERNAL,
    ],
};
