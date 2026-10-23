use crate::*;

const INTERSECT_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Intersection values or rows.",
}];
use super::super::contract::BINARY_SET_INTEGER_INPUTS;

const INTERSECT_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intersection values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in A.",
    },
];

const INTERSECT_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intersection values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in B.",
    },
];

const INTERSECT_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const INTERSECT_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const INTERSECT_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C_IA_IB,
    },
];

pub const INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.INTERSECT.LEGACY_OPTION_UNSUPPORTED",
        identifier: Some("RunMat:intersect:LegacyOptionUnsupported"),
        when: "Legacy compatibility options are requested.",
        message: "intersect: the 'legacy' behaviour is not supported",
    };

pub const INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.INTERSECT.CONFLICTING_ORDER_OPTIONS",
        identifier: Some("RunMat:intersect:ConflictingOrderOptions"),
        when: "Both 'sorted' and 'stable' options are provided.",
        message: "intersect: cannot combine 'sorted' with 'stable'",
    };

pub const INTERSECT_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.UNKNOWN_OPTION",
    identifier: Some("RunMat:intersect:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "intersect: unrecognised option",
};

pub const INTERSECT_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:intersect:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "intersect: inputs must have the same number of columns when using 'rows'",
};

pub const INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:intersect:UnsupportedInputType"),
    when: "Input values cannot be converted into supported intersect domains.",
    message: "intersect: unsupported input type",
};

pub const INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:intersect:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "intersect: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

pub const INTERSECT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.INVALID_ARGUMENT",
    identifier: Some("RunMat:intersect:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "intersect: expected string option arguments",
};

pub const INTERSECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.INTERNAL",
    identifier: Some("RunMat:intersect:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "intersect: internal operation failed",
};

const INTERSECT_ERRORS: [BuiltinErrorDescriptor; 8] = [
    INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED,
    INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS,
    INTERSECT_ERROR_UNKNOWN_OPTION,
    INTERSECT_ERROR_ROWS_COLUMN_MISMATCH,
    INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE,
    INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH,
    INTERSECT_ERROR_INVALID_ARGUMENT,
    INTERSECT_ERROR_INTERNAL,
];

pub const INTERSECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = intersect(integer_A, integer_B, options)",
        inputs: &BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const INTERSECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERSECT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTERSECT_ERRORS,
};
