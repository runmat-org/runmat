use crate::*;

const UNIQUE_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unique values or rows.",
}];

const UNIQUE_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting representatives in input A.",
    },
];

const UNIQUE_OUTPUT_C_IA_IC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting representatives in input A.",
    },
    BuiltinParamDescriptor {
        name: "ic",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices mapping each input element/row to C.",
    },
];

const UNIQUE_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const UNIQUE_INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option tokens plus the 'TreatMissingAsDistinct', logical name-value pair.",
    },
];

const UNIQUE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ic] = unique(A)",
        inputs: &UNIQUE_INPUTS_A,
        outputs: &UNIQUE_OUTPUT_C_IA_IC,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ic] = unique(A, option...)",
        inputs: &UNIQUE_INPUTS_A_OPTIONS,
        outputs: &UNIQUE_OUTPUT_C_IA_IC,
    },
];

pub const UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:unique:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "unique: the 'legacy' behaviour is not supported",
};

pub const UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:unique:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "unique: cannot combine 'sorted' with 'stable'",
};

pub const UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.UNIQUE.CONFLICTING_OCCURRENCE_OPTIONS",
        identifier: Some("RunMat:unique:ConflictingOccurrenceOptions"),
        when: "Both 'first' and 'last' options are provided.",
        message: "unique: cannot combine 'first' with 'last'",
    };

pub const UNIQUE_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.UNKNOWN_OPTION",
    identifier: Some("RunMat:unique:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "unique: unrecognised option",
};

pub const UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.ROWS_REQUIRES_2D_MATRIX",
    identifier: Some("RunMat:unique:RowsRequiresTwoDimensionalInput"),
    when: "'rows' mode is used with non-2D data.",
    message: "unique: 'rows' option requires a 2-D matrix input",
};

pub const UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:unique:UnsupportedInputType"),
    when: "Input cannot be converted into a supported unique domain.",
    message: "unique: unsupported input type",
};

pub const UNIQUE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.INVALID_ARGUMENT",
    identifier: Some("RunMat:unique:InvalidArgument"),
    when: "Option arguments or name-value pairs are malformed.",
    message: "unique: invalid option arguments",
};

pub const UNIQUE_ERROR_GPU_OPTION_COMBINATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.GPU_OPTION_COMBINATION",
    identifier: Some("RunMat:unique:GpuOptionCombination"),
    when: "A resident input specifies both set-order and occurrence options.",
    message: "unique: GPU inputs cannot combine a set-order option with 'first' or 'last'",
};

pub const UNIQUE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIQUE.INTERNAL",
    identifier: Some("RunMat:unique:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "unique: internal operation failed",
};

const UNIQUE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    UNIQUE_ERROR_LEGACY_OPTION_UNSUPPORTED,
    UNIQUE_ERROR_CONFLICTING_ORDER_OPTIONS,
    UNIQUE_ERROR_CONFLICTING_OCCURRENCE_OPTIONS,
    UNIQUE_ERROR_UNKNOWN_OPTION,
    UNIQUE_ERROR_ROWS_REQUIRES_2D_MATRIX,
    UNIQUE_ERROR_UNSUPPORTED_INPUT_TYPE,
    UNIQUE_ERROR_INVALID_ARGUMENT,
    UNIQUE_ERROR_GPU_OPTION_COMBINATION,
    UNIQUE_ERROR_INTERNAL,
];

const UNIQUE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts every real integer class and retains exact typed storage throughout host evaluation.",
    }];

pub const UNIQUE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ic] = unique(integer_A, options)",
        inputs: &UNIQUE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves A's exact integer class; ia and ic are one-based double. GPU supports integer classes through 32 bits and restores all requested outputs after typed fallback.",
    }];

pub const UNIQUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIQUE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNIQUE_ERRORS,
};
