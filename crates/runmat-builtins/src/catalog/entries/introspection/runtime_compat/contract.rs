use crate::*;

const TEXT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host text result.",
}];
const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host logical scalar.",
}];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const VERSION_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "-release, -description, or -java.",
}];
const VER_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "product",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Product name.",
    },
    BuiltinParamDescriptor {
        name: "requiredVersion",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Version to compare.",
    },
];
const VERSION_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "v = version",
        inputs: &NO_INPUTS,
        outputs: &TEXT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "v = version(option)",
        inputs: &VERSION_OPTION,
        outputs: &TEXT_OUTPUT,
    },
];
const ISDEPLOYED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isdeployed",
    inputs: &NO_INPUTS,
    outputs: &LOGICAL_OUTPUT,
}];
const VER_LESS_THAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = verLessThan(product, requiredVersion)",
    inputs: &VER_INPUTS,
    outputs: &LOGICAL_OUTPUT,
}];

pub const VERSION_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VERSION.INVALID_INPUT",
    identifier: Some("RunMat:runtimeCompat:InvalidInput"),
    when: "Argument count, text control, or option is invalid.",
    message: "version: expected zero or one supported text option",
};
pub const ISDEPLOYED_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISDEPLOYED.INVALID_INPUT",
    identifier: Some("RunMat:runtimeCompat:InvalidInput"),
    when: "Any input argument is supplied.",
    message: "isdeployed: expected no input arguments",
};
pub const VER_LESS_THAN_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VER_LESS_THAN.INVALID_INPUT",
    identifier: Some("RunMat:runtimeCompat:InvalidInput"),
    when: "A product or version control is not a text scalar.",
    message: "runtime compatibility helper: expected text scalar",
};
pub const VERSION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VERSION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[VERSION_INVALID_INPUT],
};
pub const ISDEPLOYED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISDEPLOYED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISDEPLOYED_INVALID_INPUT],
};
pub const VER_LESS_THAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VER_LESS_THAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[VER_LESS_THAN_INVALID_INPUT],
};

pub const ISDEPLOYED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "No inputs; no integer data, controls, outputs, or provider surface.",
};
pub const VERSION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only a documented text option is accepted; integer and resident inputs reject before provider access.",
};
pub const VER_LESS_THAN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes:
            "Only two text controls are accepted; native integers are not version-number aliases.",
    };
