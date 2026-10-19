use crate::*;

pub const DE2BI_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "de2bi-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "de2bi on a GPU-resident decimal array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:De2biResidentInputExtension"),
};
pub const DE2BI_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "de2bi-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "de2bi with a fixed-width integer width or base is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:De2biIntegerControlExtension"),
    };
pub const DE2BI_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "de2bi-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "de2bi on logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:De2biLogicalInputExtension"),
};
pub const DE2BI_WIDE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "de2bi-wide-integer",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "de2bi on an integer greater than 2^52 is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:De2biWideIntegerExtension"),
};
pub const DE2BI_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "de2bi-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "de2bi on an N-D decimal array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:De2biNdInputExtension"),
};
pub const DE2BI_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    DE2BI_GPU_INPUT_EXTENSION,
    DE2BI_INTEGER_CONTROL_EXTENSION,
    DE2BI_LOGICAL_INPUT_EXTENSION,
    DE2BI_WIDE_INTEGER_EXTENSION,
    DE2BI_ND_INPUT_EXTENSION,
];

const PARAM_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real nonnegative integer scalar, vector, or matrix.",
};
const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive output digit column count; [] chooses the minimum.",
};
const PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: Some("2"),
    description: "Integer base at least two.",
};
const PARAM_ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("'right-msb'"),
    description: "Digit order: right-msb or left-msb.",
};
const INPUT_D: [BuiltinParamDescriptor; 1] = [PARAM_D];
const INPUT_D_ORDER: [BuiltinParamDescriptor; 2] = [PARAM_D, PARAM_ORDER];
const INPUT_D_N: [BuiltinParamDescriptor; 2] = [PARAM_D, PARAM_N];
const INPUT_D_N_ORDER: [BuiltinParamDescriptor; 3] = [PARAM_D, PARAM_N, PARAM_ORDER];
const INPUT_D_N_P: [BuiltinParamDescriptor; 3] = [PARAM_D, PARAM_N, PARAM_P];
const INPUT_D_N_P_ORDER: [BuiltinParamDescriptor; 4] = [PARAM_D, PARAM_N, PARAM_P, PARAM_ORDER];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One row of base-P digits per input element, in the input class.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D)",
        inputs: &INPUT_D,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D, flg)",
        inputs: &INPUT_D_ORDER,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D, N)",
        inputs: &INPUT_D_N,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D, N, flg)",
        inputs: &INPUT_D_N_ORDER,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D, N, P)",
        inputs: &INPUT_D_N_P,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = de2bi(D, N, P, flg)",
        inputs: &INPUT_D_N_P_ORDER,
        outputs: &OUTPUT,
    },
];
pub const DE2BI_ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.INPUT",
    identifier: Some("RunMat:de2bi:InvalidInput"),
    when: "D is not real nonnegative numeric data with integer values.",
    message: "de2bi: D must contain real nonnegative integers",
};
pub const DE2BI_ERROR_WIDTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.WIDTH",
    identifier: Some("RunMat:de2bi:InvalidWidth"),
    when: "N is invalid or too narrow for an input value.",
    message: "de2bi: N is too small to represent an input value",
};
pub const DE2BI_ERROR_BASE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.BASE",
    identifier: Some("RunMat:de2bi:InvalidBase"),
    when: "P is not an integer scalar at least two.",
    message: "de2bi: P must be an integer greater than 1",
};
pub const DE2BI_ERROR_RESOURCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.RESOURCE",
    identifier: Some("RunMat:de2bi:ResourceLimit"),
    when: "The requested digit matrix exceeds the supported size.",
    message: "de2bi: requested output exceeds the maximum supported size",
};
pub const DE2BI_ERROR_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.ORDER",
    identifier: Some("RunMat:de2bi:InvalidOrder"),
    when: "The digit order is not right-msb or left-msb.",
    message: "de2bi: unsupported digit order",
};
pub const DE2BI_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DE2BI.PROVIDER",
    identifier: Some("RunMat:de2bi:ProviderError"),
    when: "Gathering a resident decimal array fails after admission.",
    message: "de2bi: failed to gather resident decimals",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    DE2BI_ERROR_INPUT,
    DE2BI_ERROR_WIDTH,
    DE2BI_ERROR_BASE,
    DE2BI_ERROR_RESOURCE,
    DE2BI_ERROR_ORDER,
    DE2BI_ERROR_PROVIDER,
];
pub const DE2BI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "D",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer values retain their exact bits through digit extraction; values greater than 2^52 are a separately gated RunMat extension.",
}];
const INTEGER_CONTROL: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "N",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Fixed-width width controls are decoded exactly in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "P",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Fixed-width base controls are decoded exactly in RunMat mode.",
    },
];
pub const DE2BI_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = de2bi(integer_D, N, P, flg)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The output matrix has D's numeric class. MATLAB-compatible mode follows the documented 2^52 input bound; RunMat mode supports exact wider native integers.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = de2bi(D, integer_N, integer_P, flg)",
        inputs: &INTEGER_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Both controls are exact and independently mode-gated from integer data.",
    },
];
