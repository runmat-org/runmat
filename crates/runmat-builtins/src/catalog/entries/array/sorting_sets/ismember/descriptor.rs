use crate::*;

const ISMEMBER_OUTPUT_MASK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Membership mask over A.",
}];
use super::super::contract::BINARY_SET_INTEGER_INPUTS;

const ISMEMBER_OUTPUT_MASK_LOC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Membership mask over A.",
    },
    BuiltinParamDescriptor {
        name: "loc",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First-match indices into B for each element/row in A (0 when absent).",
    },
];

const ISMEMBER_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values or rows to query.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference set of values or rows.",
    },
];

const ISMEMBER_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values or rows to query.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference set of values or rows.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option tokens: 'rows'.",
    },
];

const ISMEMBER_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "tf = ismember(A, B)",
        inputs: &ISMEMBER_INPUTS_A_B,
        outputs: &ISMEMBER_OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ismember(A, B, option...)",
        inputs: &ISMEMBER_INPUTS_A_B_OPTIONS,
        outputs: &ISMEMBER_OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[tf, loc] = ismember(A, B)",
        inputs: &ISMEMBER_INPUTS_A_B,
        outputs: &ISMEMBER_OUTPUT_MASK_LOC,
    },
    BuiltinSignatureDescriptor {
        label: "[tf, loc] = ismember(A, B, option...)",
        inputs: &ISMEMBER_INPUTS_A_B_OPTIONS,
        outputs: &ISMEMBER_OUTPUT_MASK_LOC,
    },
];

pub const ISMEMBER_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ISMEMBER.LEGACY_OPTION_UNSUPPORTED",
        identifier: Some("RunMat:ismember:LegacyOptionUnsupported"),
        when: "Legacy compatibility options are requested.",
        message: "ismember: the 'legacy' behaviour is not supported",
    };

pub const ISMEMBER_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.UNKNOWN_OPTION",
    identifier: Some("RunMat:ismember:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "ismember: unrecognised option",
};

pub const ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:ismember:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "ismember: inputs must have the same number of columns when using 'rows'",
};

pub const ISMEMBER_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:ismember:UnsupportedInputType"),
    when: "Input classes or execution residency are unsupported.",
    message: "ismember: unsupported input type",
};

pub const ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:ismember:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "ismember: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

pub const ISMEMBER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.INVALID_ARGUMENT",
    identifier: Some("RunMat:ismember:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "ismember: expected string option arguments",
};

pub const ISMEMBER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBER.INTERNAL",
    identifier: Some("RunMat:ismember:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "ismember: internal operation failed",
};

const ISMEMBER_ERRORS: [BuiltinErrorDescriptor; 7] = [
    ISMEMBER_ERROR_LEGACY_OPTION_UNSUPPORTED,
    ISMEMBER_ERROR_UNKNOWN_OPTION,
    ISMEMBER_ERROR_ROWS_COLUMN_MISMATCH,
    ISMEMBER_ERROR_UNSUPPORTED_INPUT_TYPE,
    ISMEMBER_ERROR_NUMERIC_CLASS_MISMATCH,
    ISMEMBER_ERROR_INVALID_ARGUMENT,
    ISMEMBER_ERROR_INTERNAL,
];

pub const ISMEMBER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[Lia, Locb] = ismember(integer_A, integer_B, options)",
        inputs: &BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Lia is logical and optional Locb is one-based double. Host supports all eight integer classes exactly; GPU supports integer classes through 32 bits, gathers typed fallback when needed, and restores both outputs to the owning provider.",
    }];

pub const ISMEMBER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISMEMBER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISMEMBER_ERRORS,
};
