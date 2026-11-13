use crate::*;

const SETDIFF_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values that appear in A but not in B.",
}];
use super::super::contract::BINARY_SET_INTEGER_INPUTS;

const SETDIFF_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values that appear in A but not in B.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting retained values/rows from A.",
    },
];

const SETDIFF_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const SETDIFF_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const SETDIFF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = setdiff(A, B)",
        inputs: &SETDIFF_INPUTS_A_B,
        outputs: &SETDIFF_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = setdiff(A, B, option...)",
        inputs: &SETDIFF_INPUTS_A_B_OPTIONS,
        outputs: &SETDIFF_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = setdiff(A, B)",
        inputs: &SETDIFF_INPUTS_A_B,
        outputs: &SETDIFF_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = setdiff(A, B, option...)",
        inputs: &SETDIFF_INPUTS_A_B_OPTIONS,
        outputs: &SETDIFF_OUTPUT_C_IA,
    },
];

pub const SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SETDIFF.LEGACY_OPTION_UNSUPPORTED",
        identifier: Some("RunMat:setdiff:LegacyOptionUnsupported"),
        when: "Legacy compatibility options are requested.",
        message: "setdiff: the 'legacy' behaviour is not supported",
    };

pub const SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SETDIFF.CONFLICTING_ORDER_OPTIONS",
        identifier: Some("RunMat:setdiff:ConflictingOrderOptions"),
        when: "Both 'sorted' and 'stable' options are provided.",
        message: "setdiff: cannot combine 'sorted' with 'stable'",
    };

pub const SETDIFF_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.UNKNOWN_OPTION",
    identifier: Some("RunMat:setdiff:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "setdiff: unrecognised option",
};

pub const SETDIFF_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:setdiff:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "setdiff: inputs must have the same number of columns when using 'rows'",
};

pub const SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:setdiff:UnsupportedInputType"),
    when: "Input values cannot be converted into supported setdiff domains.",
    message: "setdiff: unsupported input type",
};

pub const SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:setdiff:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "setdiff: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

pub const SETDIFF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.INVALID_ARGUMENT",
    identifier: Some("RunMat:setdiff:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "setdiff: expected string option arguments",
};

pub const SETDIFF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.INTERNAL",
    identifier: Some("RunMat:setdiff:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "setdiff: internal operation failed",
};

const SETDIFF_ERRORS: [BuiltinErrorDescriptor; 8] = [
    SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED,
    SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS,
    SETDIFF_ERROR_UNKNOWN_OPTION,
    SETDIFF_ERROR_ROWS_COLUMN_MISMATCH,
    SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE,
    SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH,
    SETDIFF_ERROR_INVALID_ARGUMENT,
    SETDIFF_ERROR_INTERNAL,
];

pub const SETDIFF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia] = setdiff(integer_A, integer_B, options)",
        inputs: &BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves A's integer class, including when B is double; ia is one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const SETDIFF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SETDIFF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SETDIFF_ERRORS,
};
