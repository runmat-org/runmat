use crate::*;

const MIN_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Minimum values.",
}];

const MIN_OUTPUT_MI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based minimum indices/origins.",
    },
];

const MIN_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
};

const MIN_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second operand for element-wise minimum.",
};

const MIN_PARAM_EMPTY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "placeholder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Empty placeholder selecting reduction-argument grammar.",
};

const MIN_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Reduction dimension selector (scalar or dimension vector).",
};

const MIN_PARAM_REDUCTION_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Reduction mode flag: \"all\" or \"linear\".",
};

const MIN_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description: "Missing-value mode: \"includenan\" or \"omitnan\".",
};

const MIN_PARAM_COMPARISON_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"ComparisonMethod\""),
    description: "Option name (currently \"ComparisonMethod\").",
};

const MIN_PARAM_COMPARISON_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"auto\""),
    description: "Comparison method: \"auto\", \"abs\"/\"magnitude\", or \"real\".",
};

const MIN_PARAM_OPTION_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value option name.",
};

const MIN_PARAM_OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value option value.",
};

const MIN_INPUTS_A: [BuiltinParamDescriptor; 1] = [MIN_PARAM_A];
const MIN_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [MIN_PARAM_A, MIN_PARAM_B];
const MIN_INPUTS_A_EMPTY_DIM: [BuiltinParamDescriptor; 3] =
    [MIN_PARAM_A, MIN_PARAM_EMPTY, MIN_PARAM_DIM];
const MIN_INPUTS_A_EMPTY_FLAG: [BuiltinParamDescriptor; 3] =
    [MIN_PARAM_A, MIN_PARAM_EMPTY, MIN_PARAM_REDUCTION_FLAG];
const MIN_INPUTS_A_EMPTY_NANFLAG: [BuiltinParamDescriptor; 3] =
    [MIN_PARAM_A, MIN_PARAM_EMPTY, MIN_PARAM_NANFLAG];
const MIN_INPUTS_A_EMPTY_COMPARISON: [BuiltinParamDescriptor; 4] = [
    MIN_PARAM_A,
    MIN_PARAM_EMPTY,
    MIN_PARAM_COMPARISON_NAME,
    MIN_PARAM_COMPARISON_VALUE,
];
const MIN_INPUTS_A_B_COMPARISON: [BuiltinParamDescriptor; 4] = [
    MIN_PARAM_A,
    MIN_PARAM_B,
    MIN_PARAM_COMPARISON_NAME,
    MIN_PARAM_COMPARISON_VALUE,
];
const MIN_INPUTS_A_EMPTY_OPTIONS: [BuiltinParamDescriptor; 4] = [
    MIN_PARAM_A,
    MIN_PARAM_EMPTY,
    MIN_PARAM_OPTION_NAME,
    MIN_PARAM_OPTION_VALUE,
];
const MIN_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 4] = [
    MIN_PARAM_A,
    MIN_PARAM_B,
    MIN_PARAM_OPTION_NAME,
    MIN_PARAM_OPTION_VALUE,
];

const MIN_SIGNATURES: [BuiltinSignatureDescriptor; 19] = [
    BuiltinSignatureDescriptor {
        label: "M = min(A)",
        inputs: &MIN_INPUTS_A,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A)",
        inputs: &MIN_INPUTS_A,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, B)",
        inputs: &MIN_INPUTS_A_B,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], dim)",
        inputs: &MIN_INPUTS_A_EMPTY_DIM,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], dim)",
        inputs: &MIN_INPUTS_A_EMPTY_DIM,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], vecdim)",
        inputs: &MIN_INPUTS_A_EMPTY_DIM,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], vecdim)",
        inputs: &MIN_INPUTS_A_EMPTY_DIM,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], \"all\")",
        inputs: &MIN_INPUTS_A_EMPTY_FLAG,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], \"all\")",
        inputs: &MIN_INPUTS_A_EMPTY_FLAG,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], \"linear\")",
        inputs: &MIN_INPUTS_A_EMPTY_FLAG,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], \"linear\")",
        inputs: &MIN_INPUTS_A_EMPTY_FLAG,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], nanflag)",
        inputs: &MIN_INPUTS_A_EMPTY_NANFLAG,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], nanflag)",
        inputs: &MIN_INPUTS_A_EMPTY_NANFLAG,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], \"ComparisonMethod\", method)",
        inputs: &MIN_INPUTS_A_EMPTY_COMPARISON,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], \"ComparisonMethod\", method)",
        inputs: &MIN_INPUTS_A_EMPTY_COMPARISON,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, B, \"ComparisonMethod\", method)",
        inputs: &MIN_INPUTS_A_B_COMPARISON,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, [], optionName, optionValue, ...)",
        inputs: &MIN_INPUTS_A_EMPTY_OPTIONS,
        outputs: &MIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = min(A, [], optionName, optionValue, ...)",
        inputs: &MIN_INPUTS_A_EMPTY_OPTIONS,
        outputs: &MIN_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = min(A, B, optionName, optionValue, ...)",
        inputs: &MIN_INPUTS_A_B_OPTIONS,
        outputs: &MIN_OUTPUT_M,
    },
];

pub const MIN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:min:InvalidArgument"),
    when: "Argument grammar, dimensions, or option names/values are invalid.",
    message: "min: invalid argument",
};

pub const MIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MIN.INVALID_INPUT",
    identifier: Some("RunMat:min:InvalidInput"),
    when: "Input values cannot be converted to supported min domains.",
    message: "min: invalid input",
};

pub const MIN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MIN.SIZE_MISMATCH",
    identifier: Some("RunMat:min:SizeMismatch"),
    when: "Element-wise operands are not broadcast-compatible.",
    message: "min: size mismatch",
};

pub const MIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MIN.INTERNAL",
    identifier: Some("RunMat:min:Internal"),
    when: "Execution fails due to gather, provider, allocation, or conversion internals.",
    message: "min: internal failure",
};

const MIN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    MIN_ERROR_INVALID_ARGUMENT,
    MIN_ERROR_INVALID_INPUT,
    MIN_ERROR_SIZE_MISMATCH,
    MIN_ERROR_INTERNAL,
];

pub const MIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MIN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MIN_ERRORS,
};

const REDUCTION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real reduction data accepts every integer storage class; complex-integer ordering remains a separately tracked conformance question.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive integer dimension selectors are decoded exactly from typed integer or integer-valued floating storage.",
    },
];

const PAIRWISE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "An integer array may pair with the same integer class or a scalar double in either operand position.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Two integer arrays must share one class; mixed integer classes and nonscalar floating arrays are rejected.",
    },
];

pub const MIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[M, I] = min(A, [], dim_or_vecdim, missingflag)",
        inputs: &REDUCTION_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Minimum values preserve the integer input class and one-based indices are double; supported provider reductions retain resident values and indices.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = min(A, B)",
        inputs: &PAIRWISE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Pairwise minimum applies compatible-size expansion and preserves the integer operand class, including exact comparison against an allowed scalar double. Missing-value flags are reduction-only.",
    },
];
