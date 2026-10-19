use crate::*;

/// Accepted property vocabulary shared by static validation and runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterPropertyCategory {
    Alpha,
    Alphanum,
    Digit,
    XDigit,
    WSpace,
    Upper,
    Lower,
    Punct,
    Control,
    Graphic,
    Print,
}

impl CharacterPropertyCategory {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "alpha" => Some(Self::Alpha),
            "alphanum" => Some(Self::Alphanum),
            "digit" => Some(Self::Digit),
            "xdigit" => Some(Self::XDigit),
            "wspace" => Some(Self::WSpace),
            "upper" => Some(Self::Upper),
            "lower" => Some(Self::Lower),
            "punct" => Some(Self::Punct),
            "cntrl" => Some(Self::Control),
            "graphic" => Some(Self::Graphic),
            "print" => Some(Self::Print),
            _ => None,
        }
    }
}
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result or a cell containing logical results.",
}];
const BASE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or numeric character codes.",
    },
    BuiltinParamDescriptor {
        name: "category",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Character-property category.",
    },
];
const OPTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or numeric character codes.",
    },
    BuiltinParamDescriptor {
        name: "category",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Character-property category.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ForceCellOutput\""),
        description: "ForceCellOutput name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-element logical or real double value equal to zero or one.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = isstrprop(text, category)",
        inputs: &BASE_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "tf = isstrprop(text, category, 'ForceCellOutput', flag)",
        inputs: &OPTION_INPUTS,
        outputs: &OUTPUTS,
    },
];
pub const ISSTRPROP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSTRPROP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:isstrprop:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "isstrprop: too many output arguments",
};
pub const ISSTRPROP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISSTRPROP_ERROR_TOO_MANY_OUTPUTS],
};
pub const ISSTRPROP_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isstrprop-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "isstrprop with an explicitly GPU-resident numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsstrpropResidentInputExtension"),
    };
pub const ISSTRPROP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISSTRPROP_RESIDENT_INPUT_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "text", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are interpreted exactly as character codes before property evaluation.",
}];
pub const ISSTRPROP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf = isstrprop(integer_text, category)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::Logical,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Integer values are decoded exactly then clamped to the BMP code-unit range; explicit resident input is mode-gated before exact gather.",
}];
