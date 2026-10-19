use crate::*;
const BUTTER_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator coefficient vector.",
}];

const BUTTER_OUTPUT_BA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
];

const BUTTER_OUTPUT_ZPK: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter zeros.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter poles.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter gain.",
    },
];

const BUTTER_OUTPUT_SS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "State matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output matrix.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Feedthrough term.",
    },
];

const BUTTER_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized digital cutoff frequency or analog cutoff frequency.",
    },
];

const BUTTER_INPUTS_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized digital cutoff frequency or analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "ftype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"low\" for scalar Wn, \"bandpass\" for two-element Wn"),
        description: "Filter type: low, high, bandpass, or stop.",
    },
];

const BUTTER_INPUTS_ANALOG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog filter-design flag.",
    },
];

const BUTTER_INPUTS_TYPE_ANALOG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "ftype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"low\" for scalar Wn, \"bandpass\" for two-element Wn"),
        description: "Filter type: low, high, bandpass, or stop.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Analog filter-design flag.",
    },
];

const BUTTER_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_SS,
    },
];

pub const BUTTER_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.ARG_COUNT",
    identifier: Some("RunMat:butter:ArgCount"),
    when: "The argument count is outside butter's supported forms.",
    message: "butter: expected two to four input arguments",
};

pub const BUTTER_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_ORDER",
    identifier: Some("RunMat:butter:InvalidOrder"),
    when: "Filter order is missing, non-finite, non-integer, less than one, or too large.",
    message: "butter: order must be a positive integer",
};

pub const BUTTER_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_FREQUENCY",
    identifier: Some("RunMat:butter:InvalidFrequency"),
    when: "Cutoff frequencies are malformed or outside the valid range.",
    message: "butter: invalid cutoff frequency",
};

pub const BUTTER_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_OPTION",
    identifier: Some("RunMat:butter:InvalidOption"),
    when: "Filter type or analog flag is malformed or unknown.",
    message: "butter: invalid option",
};

pub const BUTTER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:butter:TooManyOutputs"),
    when: "More than four outputs are requested.",
    message: "butter: too many output arguments",
};

pub const BUTTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INTERNAL",
    identifier: Some("RunMat:butter:Internal"),
    when: "Filter design or output materialization fails internally.",
    message: "butter: internal error",
};

const BUTTER_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BUTTER_ERROR_ARG_COUNT,
    BUTTER_ERROR_INVALID_ORDER,
    BUTTER_ERROR_INVALID_FREQUENCY,
    BUTTER_ERROR_INVALID_OPTION,
    BUTTER_ERROR_TOO_MANY_OUTPUTS,
    BUTTER_ERROR_INTERNAL,
];

pub const BUTTER_INTEGER_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-integer-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with a typed-integer filter order is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterIntegerOrderExtension"),
};
pub const BUTTER_LOGICAL_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-logical-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with a logical filter order is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterLogicalOrderExtension"),
};
pub const BUTTER_SINGLE_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-single-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with a single-precision filter order is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterSingleOrderExtension"),
};
pub const BUTTER_INTEGER_CUTOFF_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "butter-integer-cutoff",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "butter with typed-integer cutoff frequencies is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButterIntegerCutoffExtension"),
    };
pub const BUTTER_LOGICAL_CUTOFF_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "butter-logical-cutoff",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "butter with logical cutoff frequencies is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButterLogicalCutoffExtension"),
    };
pub const BUTTER_SINGLE_CUTOFF_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-single-cutoff",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with single-precision cutoff frequencies is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterSingleCutoffExtension"),
};
pub const BUTTER_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with resident GPU-array arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterGpuInputExtension"),
};
pub const BUTTER_OPTION_ALIAS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-option-alias",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "butter with a nonstandard filter-type alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterOptionAliasExtension"),
};
pub const BUTTER_LARGE_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "butter-order-above-500",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "butter with a filter order above the documented limit of 500 is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButterLargeOrderExtension"),
};

pub const BUTTER_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    BUTTER_INTEGER_ORDER_EXTENSION,
    BUTTER_LOGICAL_ORDER_EXTENSION,
    BUTTER_SINGLE_ORDER_EXTENSION,
    BUTTER_INTEGER_CUTOFF_EXTENSION,
    BUTTER_LOGICAL_CUTOFF_EXTENSION,
    BUTTER_SINGLE_CUTOFF_EXTENSION,
    BUTTER_GPU_INPUT_EXTENSION,
    BUTTER_OPTION_ALIAS_EXTENSION,
    BUTTER_LARGE_ORDER_EXTENSION,
];

const BUTTER_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented order datatype is double; RunMat mode additionally accepts every scalar integer class and parses it exactly into a bounded host order.",
    },
];
const BUTTER_INTEGER_CUTOFF_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "Wn",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented cutoff datatype is double; RunMat mode additionally accepts scalar or two-element integer cutoff storage before the deliberate double filter-design boundary.",
    },
];

pub const BUTTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[b,a]/[z,p,k]/[A,B,C,D] = butter(integer_n,Wn,...)",
        inputs: &BUTTER_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer n is checked exactly against positive host and documented-order bounds. Resident order input is separately gated, gathers through its owning provider in RunMat mode, and every numeric output remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[b,a]/[z,p,k]/[A,B,C,D] = butter(n,integer_Wn,...)",
        inputs: &BUTTER_INTEGER_CUTOFF_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer Wn is retained through scalar/vector validation and then explicitly materialized into the double Butterworth design domain. Resident cutoff input is separately gated and returns host double outputs.",
    },
];

pub const BUTTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUTTER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUTTER_ERRORS,
};
