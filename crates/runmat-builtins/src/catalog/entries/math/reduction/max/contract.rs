use crate::*;

const MAX_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Maximum numeric values or ordinal categorical values.",
}];

const MAX_OUTPUT_MI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum numeric values or ordinal categorical values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based maximum indices/origins.",
    },
];

const MAX_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
};

const MAX_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second operand for element-wise maximum.",
};

const MAX_PARAM_EMPTY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "placeholder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Empty placeholder selecting reduction-argument grammar.",
};

const MAX_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Reduction dimension selector (scalar or dimension vector).",
};

const MAX_PARAM_REDUCTION_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Reduction mode flag: \"all\" or \"linear\".",
};

const MAX_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description: "Missing-value mode: \"includenan\" or \"omitnan\".",
};

const MAX_PARAM_COMPARISON_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"ComparisonMethod\""),
    description: "Option name (currently \"ComparisonMethod\").",
};

const MAX_PARAM_COMPARISON_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"auto\""),
    description: "Comparison method: \"auto\", \"abs\"/\"magnitude\", or \"real\".",
};

const MAX_PARAM_OPTION_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value option name.",
};

const MAX_PARAM_OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value option value.",
};

const MAX_INPUTS_A: [BuiltinParamDescriptor; 1] = [MAX_PARAM_A];
const MAX_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [MAX_PARAM_A, MAX_PARAM_B];
const MAX_INPUTS_A_B_NANFLAG: [BuiltinParamDescriptor; 3] =
    [MAX_PARAM_A, MAX_PARAM_B, MAX_PARAM_NANFLAG];
const MAX_INPUTS_A_EMPTY_DIM: [BuiltinParamDescriptor; 3] =
    [MAX_PARAM_A, MAX_PARAM_EMPTY, MAX_PARAM_DIM];
const MAX_INPUTS_A_EMPTY_FLAG: [BuiltinParamDescriptor; 3] =
    [MAX_PARAM_A, MAX_PARAM_EMPTY, MAX_PARAM_REDUCTION_FLAG];
const MAX_INPUTS_A_EMPTY_NANFLAG: [BuiltinParamDescriptor; 3] =
    [MAX_PARAM_A, MAX_PARAM_EMPTY, MAX_PARAM_NANFLAG];
const MAX_INPUTS_A_EMPTY_COMPARISON: [BuiltinParamDescriptor; 4] = [
    MAX_PARAM_A,
    MAX_PARAM_EMPTY,
    MAX_PARAM_COMPARISON_NAME,
    MAX_PARAM_COMPARISON_VALUE,
];
const MAX_INPUTS_A_B_COMPARISON: [BuiltinParamDescriptor; 4] = [
    MAX_PARAM_A,
    MAX_PARAM_B,
    MAX_PARAM_COMPARISON_NAME,
    MAX_PARAM_COMPARISON_VALUE,
];
const MAX_INPUTS_A_EMPTY_OPTIONS: [BuiltinParamDescriptor; 4] = [
    MAX_PARAM_A,
    MAX_PARAM_EMPTY,
    MAX_PARAM_OPTION_NAME,
    MAX_PARAM_OPTION_VALUE,
];
const MAX_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 4] = [
    MAX_PARAM_A,
    MAX_PARAM_B,
    MAX_PARAM_OPTION_NAME,
    MAX_PARAM_OPTION_VALUE,
];

const MAX_SIGNATURES: [BuiltinSignatureDescriptor; 20] = [
    BuiltinSignatureDescriptor {
        label: "M = max(A)",
        inputs: &MAX_INPUTS_A,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A)",
        inputs: &MAX_INPUTS_A,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, B)",
        inputs: &MAX_INPUTS_A_B,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, B, nanflag)",
        inputs: &MAX_INPUTS_A_B_NANFLAG,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], dim)",
        inputs: &MAX_INPUTS_A_EMPTY_DIM,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], dim)",
        inputs: &MAX_INPUTS_A_EMPTY_DIM,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], vecdim)",
        inputs: &MAX_INPUTS_A_EMPTY_DIM,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], vecdim)",
        inputs: &MAX_INPUTS_A_EMPTY_DIM,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], \"all\")",
        inputs: &MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], \"all\")",
        inputs: &MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], \"linear\")",
        inputs: &MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], \"linear\")",
        inputs: &MAX_INPUTS_A_EMPTY_FLAG,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], nanflag)",
        inputs: &MAX_INPUTS_A_EMPTY_NANFLAG,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], nanflag)",
        inputs: &MAX_INPUTS_A_EMPTY_NANFLAG,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], \"ComparisonMethod\", method)",
        inputs: &MAX_INPUTS_A_EMPTY_COMPARISON,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], \"ComparisonMethod\", method)",
        inputs: &MAX_INPUTS_A_EMPTY_COMPARISON,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, B, \"ComparisonMethod\", method)",
        inputs: &MAX_INPUTS_A_B_COMPARISON,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, [], optionName, optionValue, ...)",
        inputs: &MAX_INPUTS_A_EMPTY_OPTIONS,
        outputs: &MAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = max(A, [], optionName, optionValue, ...)",
        inputs: &MAX_INPUTS_A_EMPTY_OPTIONS,
        outputs: &MAX_OUTPUT_MI,
    },
    BuiltinSignatureDescriptor {
        label: "M = max(A, B, optionName, optionValue, ...)",
        inputs: &MAX_INPUTS_A_B_OPTIONS,
        outputs: &MAX_OUTPUT_M,
    },
];

pub const MAX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAX.INVALID_ARGUMENT",
    identifier: Some("RunMat:max:InvalidArgument"),
    when: "Argument grammar, dimensions, or option names/values are invalid.",
    message: "max: invalid argument",
};

pub const MAX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAX.INVALID_INPUT",
    identifier: Some("RunMat:max:InvalidInput"),
    when: "Input values cannot be converted to supported max domains.",
    message: "max: invalid input",
};

pub const MAX_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAX.SIZE_MISMATCH",
    identifier: Some("RunMat:max:SizeMismatch"),
    when: "Element-wise operands are not broadcast-compatible.",
    message: "max: size mismatch",
};

pub const MAX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAX.INTERNAL",
    identifier: Some("RunMat:max:Internal"),
    when: "Execution fails due to gather, provider, allocation, or conversion internals.",
    message: "max: internal failure",
};

// The ordinal-categorical route is owned by the table service and executes
// before numeric max admission. Preserve its declared identifiers rather than
// reclassifying those failures as numeric max diagnostics.
const MAX_ERRORS: [BuiltinErrorDescriptor; 6] = [
    MAX_ERROR_INVALID_ARGUMENT,
    MAX_ERROR_INVALID_INPUT,
    MAX_ERROR_SIZE_MISMATCH,
    MAX_ERROR_INTERNAL,
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_VARIABLE,
];

pub const MAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAX_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MAX_ERRORS,
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

pub const MAX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[M, I] = max(A, [], dim_or_vecdim, missingflag)",
        inputs: &REDUCTION_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Maximum values preserve the integer input class and one-based indices are double; supported provider reductions retain resident values and indices.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = max(A, B, missingflag)",
        inputs: &PAIRWISE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Pairwise maximum applies compatible-size expansion, accepts include/omit-NaN flags, and preserves the integer operand class, including exact comparison against an allowed scalar double.",
    },
];
