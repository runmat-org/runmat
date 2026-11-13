use crate::*;

pub const FIND_ELEMENT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "findElement accepts a scalar htmlTree object and a textual CSS selector. All eight integer classes, logical and complex values, and resident numeric handles reject without numeric-to-object conversion or provider access.",
    };

const OUT_SUBTREES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "subtrees",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching htmlTree subtrees.",
}];

const IN_TREE_SELECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tree",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "htmlTree object or cell array of htmlTree objects.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "CSS selector.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HTML.INVALID_INPUT",
    identifier: Some("RunMat:html:InvalidInput"),
    when: "Inputs are not a supported htmlTree or extractHTMLText form.",
    message: "HTML Text Analytics helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [ERROR_INVALID_INPUT, FIND_ELEMENT_ERROR_TOO_MANY_OUTPUTS];

pub const FIND_ELEMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "subtrees = findElement(tree, selector)",
        inputs: &IN_TREE_SELECTOR,
        outputs: &OUT_SUBTREES,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FIND_ELEMENT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIND_ELEMENT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:findElement:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "findElement: too many output arguments",
};
