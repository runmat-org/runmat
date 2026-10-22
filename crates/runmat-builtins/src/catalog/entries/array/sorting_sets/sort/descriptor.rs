use crate::*;

const SORT_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sorted values.",
}];

const SORT_OUTPUT_BI: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sorted values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based index permutation for each sorted slice.",
    },
];

const SORT_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const SORT_INPUTS_A_ARG1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token ('ascend'/'descend').",
    },
];

const SORT_INPUTS_A_ARG1_ARG2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector, placeholder, or direction token.",
    },
    BuiltinParamDescriptor {
        name: "arg2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension selector or direction token.",
    },
];

const SORT_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
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

const SORT_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional dimension/direction arguments.",
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
        description: "Missing-value placement: 'auto', 'first', or 'last'.",
    },
];

const SORT_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "B = sort(A)",
        inputs: &SORT_INPUTS_A,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, arg1)",
        inputs: &SORT_INPUTS_A_ARG1,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, arg1, arg2)",
        inputs: &SORT_INPUTS_A_ARG1_ARG2,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORT_INPUTS_COMPARISON_METHOD,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = sort(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORT_INPUTS_MISSING_PLACEMENT,
        outputs: &SORT_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A)",
        inputs: &SORT_INPUTS_A,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, arg1)",
        inputs: &SORT_INPUTS_A_ARG1,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, arg1, arg2)",
        inputs: &SORT_INPUTS_A_ARG1_ARG2,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, ..., \"ComparisonMethod\", method)",
        inputs: &SORT_INPUTS_COMPARISON_METHOD,
        outputs: &SORT_OUTPUT_BI,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = sort(A, ..., \"MissingPlacement\", placement)",
        inputs: &SORT_INPUTS_MISSING_PLACEMENT,
        outputs: &SORT_OUTPUT_BI,
    },
];

pub const SORT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INVALID_DIMENSION",
    identifier: Some("RunMat:sort:InvalidDimension"),
    when: "Dimension argument is non-positive, non-integer, or otherwise invalid.",
    message: "sort: invalid dimension argument",
};

pub const SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORT.COMPARISON_METHOD_REQUIRES_STRING",
        identifier: Some("RunMat:sort:ComparisonMethodRequiresString"),
        when: "ComparisonMethod option value is not string-like.",
        message: "sort: 'ComparisonMethod' requires a string value",
    };

pub const SORT_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.COMPARISON_METHOD_UNKNOWN",
    identifier: Some("RunMat:sort:ComparisonMethodUnknown"),
    when: "ComparisonMethod option value is not one of 'auto'/'real'/'abs'.",
    message: "sort: unsupported ComparisonMethod",
};

pub const SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.SORT.MISSINGPLACEMENT_REQUIRES_STRING",
        identifier: Some("RunMat:sort:MissingPlacementRequiresString"),
        when: "MissingPlacement option value is not string-like.",
        message: "sort: 'MissingPlacement' requires a string value",
    };

pub const SORT_ERROR_MISSINGPLACEMENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.MISSINGPLACEMENT_UNKNOWN",
    identifier: Some("RunMat:sort:MissingPlacementUnknown"),
    when: "MissingPlacement option value is not one of 'auto'/'first'/'last'.",
    message: "sort: unsupported MissingPlacement",
};

pub const SORT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INVALID_ARGUMENT",
    identifier: Some("RunMat:sort:InvalidArgument"),
    when: "Parser encounters invalid or unrecognized option/value arguments.",
    message: "sort: invalid argument sequence",
};

pub const SORT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SORT.INTERNAL",
    identifier: Some("RunMat:sort:Internal"),
    when: "Internal conversion, allocation, or provider result construction fails.",
    message: "sort: internal operation failed",
};

const SORT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SORT_ERROR_INVALID_DIMENSION,
    SORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
    SORT_ERROR_COMPARISON_METHOD_UNKNOWN,
    SORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING,
    SORT_ERROR_MISSINGPLACEMENT_UNKNOWN,
    SORT_ERROR_INVALID_ARGUMENT,
    SORT_ERROR_INTERNAL,
];

const SORT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sortable input domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive integer-scalar dimension accepts every integer class and integer-valued scalar double.",
    },
];

pub const SORT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = sort(integer_A, integer_dim, direction, options)",
        inputs: &SORT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident integer input uses exact typed gather fallback and restores both outputs to the owning provider.",
    }];

pub const SORT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SORT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SORT_ERRORS,
};
