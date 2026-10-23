use crate::*;

const ARGSORT_OUTPUT_I: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based permutation indices that sort each slice.",
}];

const ARGSORT_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const ARGSORT_INPUTS_A_ARG1: [BuiltinParamDescriptor; 2] = [
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
        description: "Dimension selector or direction token.",
    },
];

const ARGSORT_INPUTS_A_ARG1_ARG2: [BuiltinParamDescriptor; 3] = [
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

const ARGSORT_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
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

const ARGSORT_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
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
        description: "Requested NaN placement option (currently unsupported).",
    },
];

const ARGSORT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "I = argsort(A)",
        inputs: &ARGSORT_INPUTS_A,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, arg1)",
        inputs: &ARGSORT_INPUTS_A_ARG1,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, arg1, arg2)",
        inputs: &ARGSORT_INPUTS_A_ARG1_ARG2,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, ..., \"ComparisonMethod\", method)",
        inputs: &ARGSORT_INPUTS_COMPARISON_METHOD,
        outputs: &ARGSORT_OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = argsort(A, ..., \"MissingPlacement\", placement)",
        inputs: &ARGSORT_INPUTS_MISSING_PLACEMENT,
        outputs: &ARGSORT_OUTPUT_I,
    },
];

pub const ARGSORT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.INVALID_DIMENSION",
    identifier: Some("RunMat:sort:InvalidDimension"),
    when: "Dimension argument is non-positive, non-integer, or otherwise invalid.",
    message: "sort: invalid dimension argument",
};

pub const ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGSORT.COMPARISON_METHOD_REQUIRES_STRING",
        identifier: Some("RunMat:sort:ComparisonMethodRequiresString"),
        when: "ComparisonMethod option value is not string-like.",
        message: "sort: 'ComparisonMethod' requires a string value",
    };

pub const ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGSORT.COMPARISON_METHOD_UNKNOWN",
        identifier: Some("RunMat:sort:ComparisonMethodUnknown"),
        when: "ComparisonMethod option value is not one of 'auto'/'real'/'abs'.",
        message: "sort: unsupported ComparisonMethod",
    };

const ARGSORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGSORT.MISSINGPLACEMENT_REQUIRES_STRING",
        identifier: Some("RunMat:sort:MissingPlacementRequiresString"),
        when: "MissingPlacement option value is not string-like.",
        message: "sort: 'MissingPlacement' requires a string value",
    };

pub const ARGSORT_ERROR_MISSINGPLACEMENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.MISSINGPLACEMENT_UNKNOWN",
    identifier: Some("RunMat:sort:MissingPlacementUnknown"),
    when: "MissingPlacement option value is not one of 'auto'/'first'/'last'.",
    message: "sort: unsupported MissingPlacement",
};

pub const ARGSORT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGSORT.INVALID_ARGUMENT",
    identifier: Some("RunMat:sort:InvalidArgument"),
    when: "Parser encounters invalid or unrecognized option/value arguments.",
    message: "sort: invalid argument sequence",
};

const ARGSORT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    ARGSORT_ERROR_INVALID_DIMENSION,
    ARGSORT_ERROR_COMPARISON_METHOD_REQUIRES_STRING,
    ARGSORT_ERROR_COMPARISON_METHOD_UNKNOWN,
    ARGSORT_ERROR_MISSINGPLACEMENT_REQUIRES_STRING,
    ARGSORT_ERROR_MISSINGPLACEMENT_UNKNOWN,
    ARGSORT_ERROR_INVALID_ARGUMENT,
];

pub const ARGSORT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARGSORT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ARGSORT_ERRORS,
};

pub const ARGSORT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::AliasOf,
    canonical_builtin: Some("sort"),
    notes: "argsort delegates to sort and returns its one-based permutation-index output, so sort's documented exact integer input, dimension, and resident-provider contract is canonical.",
};
