use crate::*;

pub const BI2DE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bi2de-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bi2de on a GPU-resident digit matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Bi2deResidentInputExtension"),
};
pub const BI2DE_INTEGER_BASE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bi2de-integer-base",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bi2de with a fixed-width integer base argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Bi2deIntegerBaseExtension"),
};
pub const BI2DE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [BI2DE_GPU_INPUT_EXTENSION, BI2DE_INTEGER_BASE_EXTENSION];

const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric or logical digit row/vector/matrix.",
};
const PARAM_ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("'right-msb'"),
    description: "Digit order: right-msb or left-msb.",
};
const PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: Some("2"),
    description: "Integer base at least two.",
};
const DIGITS: [BuiltinParamDescriptor; 1] = [PARAM_B];
const DIGITS_ORDER: [BuiltinParamDescriptor; 2] = [PARAM_B, PARAM_ORDER];
const DIGITS_BASE: [BuiltinParamDescriptor; 2] = [PARAM_B, PARAM_P];
const DIGITS_BASE_ORDER: [BuiltinParamDescriptor; 3] = [PARAM_B, PARAM_P, PARAM_ORDER];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One decimal value for each digit row.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "D = bi2de(B)",
        inputs: &DIGITS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = bi2de(B, flg)",
        inputs: &DIGITS_ORDER,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = bi2de(B, P)",
        inputs: &DIGITS_BASE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = bi2de(B, P, flg)",
        inputs: &DIGITS_BASE_ORDER,
        outputs: &OUTPUT,
    },
];
pub const BI2DE_ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BI2DE.INPUT",
    identifier: Some("RunMat:bi2de:InvalidInput"),
    when: "B is not a real numeric/logical 2-D digit array, or a digit is invalid.",
    message: "bi2de: B must contain valid real digits",
};
pub const BI2DE_ERROR_BASE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BI2DE.BASE",
    identifier: Some("RunMat:bi2de:InvalidBase"),
    when: "P is not an integer scalar at least two.",
    message: "bi2de: P must be an integer greater than 1",
};
pub const BI2DE_ERROR_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BI2DE.ORDER",
    identifier: Some("RunMat:bi2de:InvalidOrder"),
    when: "The digit order is not right-msb or left-msb.",
    message: "bi2de: unsupported digit order",
};
pub const BI2DE_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BI2DE.RANGE",
    identifier: Some("RunMat:bi2de:ExactRange"),
    when: "A converted result exceeds the exact supported decimal range.",
    message: "bi2de: result exceeds the maximum exact integer supported by RunMat",
};
pub const BI2DE_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BI2DE.PROVIDER",
    identifier: Some("RunMat:bi2de:ProviderError"),
    when: "Gathering a resident digit matrix fails after admission.",
    message: "bi2de: failed to gather resident digits",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    BI2DE_ERROR_INPUT,
    BI2DE_ERROR_BASE,
    BI2DE_ERROR_ORDER,
    BI2DE_ERROR_RANGE,
    BI2DE_ERROR_PROVIDER,
];
pub const BI2DE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DIGITS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "B",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer digits are read from native storage and checked against the base before conversion.",
}];
const INTEGER_BASE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "P",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The documented base uses single or double; RunMat additionally accepts exact fixed-width integer controls.",
}];
pub const BI2DE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = bi2de(integer_B, P, flg)",
        inputs: &INTEGER_DIGITS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The output retains the integer input class when it fits; otherwise it widens in the same signedness. Results above 2^52 reject rather than round through double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "D = bi2de(B, integer_P, flg)",
        inputs: &INTEGER_BASE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The fixed-width base is decoded exactly and admitted only in RunMat mode.",
    },
];
