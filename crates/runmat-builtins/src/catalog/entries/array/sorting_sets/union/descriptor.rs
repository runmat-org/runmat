use crate::*;

const UNION_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Union values or rows.",
}];
use super::super::contract::BINARY_SET_INTEGER_INPUTS;

const UNION_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Union values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from A.",
    },
];

const UNION_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Union values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from B.",
    },
];

const UNION_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const UNION_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const UNION_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C_IA_IB,
    },
];

pub const UNION_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:union:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "union: the 'legacy' behaviour is not supported",
};

pub const UNION_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:union:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "union: cannot combine 'sorted' with 'stable'",
};

pub const UNION_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.UNKNOWN_OPTION",
    identifier: Some("RunMat:union:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "union: unrecognised option",
};

pub const UNION_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:union:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "union: inputs must have the same number of columns when using 'rows'",
};

pub const UNION_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:union:UnsupportedInputType"),
    when: "Input values cannot be converted into supported union domains.",
    message: "union: unsupported input type",
};

pub const UNION_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:union:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "union: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

pub const UNION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.INVALID_ARGUMENT",
    identifier: Some("RunMat:union:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "union: expected string option arguments",
};

pub const UNION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.INTERNAL",
    identifier: Some("RunMat:union:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "union: internal operation failed",
};

const UNION_ERRORS: [BuiltinErrorDescriptor; 8] = [
    UNION_ERROR_LEGACY_OPTION_UNSUPPORTED,
    UNION_ERROR_CONFLICTING_ORDER_OPTIONS,
    UNION_ERROR_UNKNOWN_OPTION,
    UNION_ERROR_ROWS_COLUMN_MISMATCH,
    UNION_ERROR_UNSUPPORTED_INPUT_TYPE,
    UNION_ERROR_NUMERIC_CLASS_MISMATCH,
    UNION_ERROR_INVALID_ARGUMENT,
    UNION_ERROR_INTERNAL,
];

pub const UNION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = union(integer_A, integer_B, options)",
        inputs: &BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const UNION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNION_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNION_ERRORS,
};
