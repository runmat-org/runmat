use crate::*;

pub const NANMAX_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "nanmax-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "nanmax with a typed-integer data or control input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NanmaxTypedIntegerInputExtension"),
};

pub const NANMAX_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NANMAX_INTEGER_EXTENSION];

const NANMAX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_B_or_dimension",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer data, pairwise operands, and dimension controls are accepted only with the nanmax typed-integer-input extension; documented single- and double-valued forms remain available in MATLAB-compatible mode.",
    }];

pub const NANMAX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "M = nanmax(A, args...) or nanmax(A, B)",
        inputs: &NANMAX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmax with exact typed-integer reductions and compatible pairwise forms while preserving all eight integer classes; omit-NaN resident reductions gather because the max provider path does not expose NaN-mode selection.",
    }];

const NANMAX_MAX_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Maximum values.",
}];

const NANMAX_MAX_OUTPUT_MI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based maximum indices/origins.",
    },
];

const NANMAX_MAX_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
};

const NANMAX_MAX_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second operand for element-wise maximum.",
};

const NANMAX_MAX_PARAM_EMPTY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "placeholder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Empty placeholder selecting reduction-argument grammar.",
};

const NANMAX_MAX_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Reduction dimension selector (scalar or dimension vector).",
};

const NANMAX_MAX_PARAM_REDUCTION_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"all\""),
    description: "Reduction mode flag: \"all\" or \"linear\".",
};

const NANMAX_MAX_PARAM_COMPARISON_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"ComparisonMethod\""),
    description: "Option name (currently \"ComparisonMethod\").",
};

const NANMAX_MAX_PARAM_COMPARISON_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"auto\""),
    description: "Comparison method: \"auto\", \"abs\"/\"magnitude\", or \"real\".",
};

const NANMAX_MAX_INPUTS_A_EMPTY_FLAG: [BuiltinParamDescriptor; 3] = [
    NANMAX_MAX_PARAM_A,
    NANMAX_MAX_PARAM_EMPTY,
    NANMAX_MAX_PARAM_REDUCTION_FLAG,
];

const NANMAX_MAX_INPUTS_A_EMPTY_COMPARISON: [BuiltinParamDescriptor; 4] = [
    NANMAX_MAX_PARAM_A,
    NANMAX_MAX_PARAM_EMPTY,
    NANMAX_MAX_PARAM_COMPARISON_NAME,
    NANMAX_MAX_PARAM_COMPARISON_VALUE,
];

const NANMAX_MAX_INPUTS_A_B_COMPARISON: [BuiltinParamDescriptor; 4] = [
    NANMAX_MAX_PARAM_A,
    NANMAX_MAX_PARAM_B,
    NANMAX_MAX_PARAM_COMPARISON_NAME,
    NANMAX_MAX_PARAM_COMPARISON_VALUE,
];

const NANMAX_INPUTS_A: [BuiltinParamDescriptor; 1] = [NANMAX_MAX_PARAM_A];

const NANMAX_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [NANMAX_MAX_PARAM_A, NANMAX_MAX_PARAM_B];

const NANMAX_INPUTS_A_EMPTY_DIM: [BuiltinParamDescriptor; 3] = [
    NANMAX_MAX_PARAM_A,
    NANMAX_MAX_PARAM_EMPTY,
    NANMAX_MAX_PARAM_DIM,
];

const NANMAX_SIGNATURES: [BuiltinSignatureDescriptor; 14] = [
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A)",
        inputs: &NANMAX_INPUTS_A,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A)",
        inputs: &NANMAX_INPUTS_A,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, B)",
        inputs: &NANMAX_INPUTS_A_B,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, [], dim)",
        inputs: &NANMAX_INPUTS_A_EMPTY_DIM,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A, [], dim)",
        inputs: &NANMAX_INPUTS_A_EMPTY_DIM,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, [], vecdim)",
        inputs: &NANMAX_INPUTS_A_EMPTY_DIM,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A, [], vecdim)",
        inputs: &NANMAX_INPUTS_A_EMPTY_DIM,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, [], \"all\")",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A, [], \"all\")",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, [], \"linear\")",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A, [], \"linear\")",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, [], \"ComparisonMethod\", method)",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_COMPARISON,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = nanmax(A, [], \"ComparisonMethod\", method)",
        inputs: &NANMAX_MAX_INPUTS_A_EMPTY_COMPARISON,
        outputs: &NANMAX_MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = nanmax(A, B, \"ComparisonMethod\", method)",
        inputs: &NANMAX_MAX_INPUTS_A_B_COMPARISON,
        outputs: &NANMAX_MAX_OUTPUT_M,
    },
];

pub const NANMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NANMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NANMAX_ERRORS,
};

pub const NANMAX_ERRORS: [BuiltinErrorDescriptor; 5] = [
    MAX_ERROR_INVALID_ARGUMENT,
    MAX_ERROR_INVALID_INPUT,
    MAX_ERROR_SIZE_MISMATCH,
    MAX_ERROR_INTERNAL,
    NANMAX_ERROR_TOO_MANY_OUTPUTS,
];

pub const NANMAX_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NANMAX.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:nanmax:TooManyOutputs"),
    when: "More outputs are requested than this public form returns.",
    message: "nanmax: too many output arguments",
};
