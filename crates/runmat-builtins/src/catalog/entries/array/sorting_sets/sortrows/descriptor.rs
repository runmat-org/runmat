use crate::*;

const SORTROWS_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sorted input rows.",
}];

const SORTROWS_OUTPUT_BI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sorted input rows.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation indices mapping sorted rows to original rows.",
    },
];

const SORTROWS_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to sort by rows.",
}];

const SORTROWS_INPUTS_A_COLUMNS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "column",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column specification vector (negative entries request descending order).",
    },
];

const SORTROWS_INPUTS_A_DIRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ascend\""),
        description: "Global row direction override: 'ascend' or 'descend'.",
    },
];

const SORTROWS_INPUTS_A_COLUMNS_DIRECTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "column",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column specification vector (negative entries request descending order).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ascend\""),
        description: "Global row direction override: 'ascend' or 'descend'.",
    },
];

const SORTROWS_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional column and direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ComparisonMethod\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "Comparison method: 'auto', 'real', or 'abs'.",
    },
];

const SORTROWS_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to sort by rows.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional column and direction arguments.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"MissingPlacement\""),
        description: "Name-value option key.",
    },
    BuiltinParamDescriptor {
        name: "placement",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"auto\""),
        description: "NaN placement policy: 'auto', 'first', or 'last'.",
    },
];

const SORTROWS_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A)",
        inputs: &SORTROWS_INPUTS_A,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, column)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, direction)",
        inputs: &SORTROWS_INPUTS_A_DIRECTION,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, column, direction)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS_DIRECTION,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORTROWS_INPUTS_COMPARISON_METHOD,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sortrows(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORTROWS_INPUTS_MISSING_PLACEMENT,
        outputs: &SORTROWS_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A)",
        inputs: &SORTROWS_INPUTS_A,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, column)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, direction)",
        inputs: &SORTROWS_INPUTS_A_DIRECTION,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, column, direction)",
        inputs: &SORTROWS_INPUTS_A_COLUMNS_DIRECTION,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORTROWS_INPUTS_COMPARISON_METHOD,
        outputs: &SORTROWS_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sortrows(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORTROWS_INPUTS_MISSING_PLACEMENT,
        outputs: &SORTROWS_OUTPUT_BI,
    },
];

pub const SORTROWS_ERROR_INVALID_COLUMN_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INVALID_COLUMN_INDEX",
    identifier: Some("RunMat:sortrows:InvalidColumnIndex"),
    when: "Column specification indices are out of range, zero, or otherwise invalid.",
    message: "sortrows: invalid column index",
};

pub const SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORTROWS.MISSING_PLACEMENT_UNKNOWN",
        identifier: Some("RunMat:sortrows:MissingPlacementUnknown"),
        when: "MissingPlacement option value is unsupported.",
        message: "sortrows: unsupported MissingPlacement value",
    };

pub const SORTROWS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INVALID_ARGUMENT",
    identifier: Some("RunMat:sortrows:InvalidArgument"),
    when: "Option parsing receives invalid argument kinds or malformed name-value pairs.",
    message: "sortrows: invalid argument",
};

pub const SORTROWS_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORTROWS.COMPARISON_METHOD_UNKNOWN",
        identifier: Some("RunMat:sortrows:ComparisonMethodUnknown"),
        when: "ComparisonMethod option value is unsupported.",
        message: "sortrows: unsupported ComparisonMethod value",
    };

pub const SORTROWS_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:sortrows:UnsupportedInputType"),
    when: "Input cannot be converted to numeric, logical, complex, or char matrix domain.",
    message: "sortrows: unsupported input type",
};

pub const SORTROWS_ERROR_MATRIX_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.MATRIX_REQUIRED",
    identifier: Some("RunMat:sortrows:MatrixRequired"),
    when: "Input has rank greater than 2 where matrix input is required.",
    message: "sortrows: input must be a 2-D matrix",
};

pub const SORTROWS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORTROWS.INTERNAL",
    identifier: Some("RunMat:sortrows:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "sortrows: internal operation failed",
};

const SORTROWS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SORTROWS_ERROR_INVALID_COLUMN_INDEX,
    SORTROWS_ERROR_MISSING_PLACEMENT_UNKNOWN,
    SORTROWS_ERROR_INVALID_ARGUMENT,
    SORTROWS_ERROR_COMPARISON_METHOD_UNKNOWN,
    SORTROWS_ERROR_UNSUPPORTED_INPUT_TYPE,
    SORTROWS_ERROR_MATRIX_REQUIRED,
    SORTROWS_ERROR_INTERNAL,
];

const SORTROWS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sortable matrix domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "column",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonzero integer column scalar or vector accepts every integer class and integer-valued double.",
    },
];

pub const SORTROWS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = sortrows(integer_A, integer_columns, direction, options)",
        inputs: &SORTROWS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-row order; optional I is one-based double. Resident integer input uses exact typed gather fallback and restores both outputs to the owning provider.",
    }];

pub const SORTROWS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SORTROWS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SORTROWS_ERRORS,
};
