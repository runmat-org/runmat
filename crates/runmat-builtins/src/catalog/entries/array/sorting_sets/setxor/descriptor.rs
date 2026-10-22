use crate::*;

const SETXOR_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values or rows that appear in exactly one input.",
}];
use super::super::contract::BINARY_SET_INTEGER_INPUTS;

const SETXOR_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values or rows that appear in exactly one input.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting values or rows from A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting values or rows from B.",
    },
];

const SETXOR_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
];

const SETXOR_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option tokens: 'rows'|'sorted'|'stable'.",
    },
];

const SETXOR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = setxor(A, B)",
        inputs: &SETXOR_INPUTS_A_B,
        outputs: &SETXOR_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = setxor(A, B, option...)",
        inputs: &SETXOR_INPUTS_A_B_OPTIONS,
        outputs: &SETXOR_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = setxor(A, B)",
        inputs: &SETXOR_INPUTS_A_B,
        outputs: &SETXOR_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = setxor(A, B, option...)",
        inputs: &SETXOR_INPUTS_A_B_OPTIONS,
        outputs: &SETXOR_OUTPUT_C_IA_IB,
    },
];

pub const SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:setxor:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "setxor: the 'legacy' behaviour is not supported",
};

pub const SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:setxor:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "setxor: cannot combine 'sorted' with 'stable'",
};

pub const SETXOR_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.UNKNOWN_OPTION",
    identifier: Some("RunMat:setxor:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "setxor: unrecognised option",
};

pub const SETXOR_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:setxor:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "setxor: inputs must have the same number of columns when using 'rows'",
};

pub const SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:setxor:UnsupportedInputType"),
    when: "Input values cannot be converted into supported setxor domains.",
    message: "setxor: unsupported input type",
};

pub const SETXOR_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:setxor:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "setxor: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

pub const SETXOR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:setxor:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "setxor: expected string option arguments",
};

pub const SETXOR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:setxor:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "setxor: too many output arguments",
};

pub const SETXOR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.INTERNAL",
    identifier: Some("RunMat:setxor:Internal"),
    when: "Internal conversion, allocation, or provider decode fails.",
    message: "setxor: internal operation failed",
};

const SETXOR_ERRORS: [BuiltinErrorDescriptor; 9] = [
    SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED,
    SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS,
    SETXOR_ERROR_UNKNOWN_OPTION,
    SETXOR_ERROR_ROWS_COLUMN_MISMATCH,
    SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
    SETXOR_ERROR_NUMERIC_CLASS_MISMATCH,
    SETXOR_ERROR_INVALID_ARGUMENT,
    SETXOR_ERROR_TOO_MANY_OUTPUTS,
    SETXOR_ERROR_INTERNAL,
];

pub const SETXOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = setxor(integer_A, integer_B, options)",
        inputs: &BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const SETXOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SETXOR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SETXOR_ERRORS,
};
