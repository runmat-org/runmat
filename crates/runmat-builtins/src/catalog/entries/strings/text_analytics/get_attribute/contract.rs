use crate::*;

pub const GET_ATTRIBUTE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "htmlTree.getAttribute accepts host htmlTree objects and a host text attribute name only. All eight integer classes and provider-resident numeric values reject without implicit conversion, gather, or provider access.",
    };

const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text.",
}];

const IN_TREE_ATTR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tree",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "htmlTree object or cell array of htmlTree objects.",
    },
    BuiltinParamDescriptor {
        name: "attr",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute name.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HTML.INVALID_INPUT",
    identifier: Some("RunMat:html:InvalidInput"),
    when: "Inputs are not a supported htmlTree or extractHTMLText form.",
    message: "HTML Text Analytics helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [ERROR_INVALID_INPUT, GET_ATTRIBUTE_ERROR_TOO_MANY_OUTPUTS];

pub const GET_ATTRIBUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "str = getAttribute(tree, attr)",
        inputs: &IN_TREE_ATTR,
        outputs: &OUT_TEXT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GET_ATTRIBUTE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET_ATTRIBUTE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:getAttribute:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "getAttribute: too many output arguments",
};
