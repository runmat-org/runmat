use crate::*;

const ISSORTED_OUTPUT_TF: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when data is already sorted by requested criteria.",
}];

const ISSORTED_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const ISSORTED_INPUTS_A_ARG1: [BuiltinParamDescriptor; 2] = [
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
        description: "Dimension selector or direction token (including 'rows').",
    },
];

const ISSORTED_INPUTS_A_ARG1_ARG2: [BuiltinParamDescriptor; 3] = [
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
        description: "Dimension selector or direction token (including 'rows').",
    },
    BuiltinParamDescriptor {
        name: "arg2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Direction token or additional mode selector.",
    },
];

const ISSORTED_INPUTS_COMPARISON_METHOD: [BuiltinParamDescriptor; 4] = [
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
        description: "Optional dimension/direction/rows arguments.",
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

const ISSORTED_INPUTS_MISSING_PLACEMENT: [BuiltinParamDescriptor; 4] = [
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
        description: "Optional dimension/direction/rows arguments.",
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
        description: "Missing placement policy: 'auto', 'first', or 'last'.",
    },
];

const ISSORTED_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "tf = issorted(A)",
        inputs: &ISSORTED_INPUTS_A,
        outputs: &ISSORTED_OUTPUT_TF,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issorted(A, arg1)",
        inputs: &ISSORTED_INPUTS_A_ARG1,
        outputs: &ISSORTED_OUTPUT_TF,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issorted(A, arg1, arg2)",
        inputs: &ISSORTED_INPUTS_A_ARG1_ARG2,
        outputs: &ISSORTED_OUTPUT_TF,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issorted(A, ..., \"ComparisonMethod\", method)",
        inputs: &ISSORTED_INPUTS_COMPARISON_METHOD,
        outputs: &ISSORTED_OUTPUT_TF,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issorted(A, ..., \"MissingPlacement\", placement)",
        inputs: &ISSORTED_INPUTS_MISSING_PLACEMENT,
        outputs: &ISSORTED_OUTPUT_TF,
    },
];

pub const ISSORTED_ERROR_ROWS_REQUIRES_2D: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTED.ROWS_REQUIRES_2D",
    identifier: Some("RunMat:issorted:RowsRequiresTwoDimensionalInput"),
    when: "'rows' mode is used with non-2D input.",
    message: "issorted: 'rows' expects a 2-D matrix",
};

pub const ISSORTED_ERROR_STRING_COMPARISON_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ISSORTED.STRING_COMPARISON_UNSUPPORTED",
        identifier: Some("RunMat:issorted:StringComparisonMethodUnsupported"),
        when: "ComparisonMethod is used with string arrays.",
        message: "issorted: 'ComparisonMethod' is not supported for string arrays",
    };

pub const ISSORTED_ERROR_DUPLICATE_DIRECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTED.DUPLICATE_DIRECTION",
    identifier: Some("RunMat:issorted:DuplicateDirection"),
    when: "Multiple direction tokens are provided.",
    message: "issorted: sorting direction specified more than once",
};

pub const ISSORTED_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTED.INVALID_ARGUMENT",
    identifier: Some("RunMat:issorted:InvalidArgument"),
    when: "Parser encounters invalid or unrecognized option/value arguments.",
    message: "issorted: invalid argument sequence",
};

pub const ISSORTED_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTED.INVALID_INPUT",
    identifier: Some("RunMat:issorted:InvalidInput"),
    when: "Input type cannot be normalized into a supported sortable representation.",
    message: "issorted: invalid input type",
};

pub const ISSORTED_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTED.INTERNAL",
    identifier: Some("RunMat:issorted:Internal"),
    when: "Internal conversion/allocation paths fail.",
    message: "issorted: internal operation failed",
};

const ISSORTED_ERRORS: [BuiltinErrorDescriptor; 6] = [
    ISSORTED_ERROR_ROWS_REQUIRES_2D,
    ISSORTED_ERROR_STRING_COMPARISON_UNSUPPORTED,
    ISSORTED_ERROR_DUPLICATE_DIRECTION,
    ISSORTED_ERROR_INVALID_ARGUMENT,
    ISSORTED_ERROR_INVALID_INPUT,
    ISSORTED_ERROR_INTERNAL,
];

pub const ISSORTED_GPU_NONVECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issorted-gpu-nonvector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issorted with a nonvector resident GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IssortedGpuNonvectorExtension"),
    };

pub const ISSORTED_GPU_MISSING_PLACEMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issorted-gpu-missing-placement",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issorted with a non-auto MissingPlacement value on GPU is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IssortedGpuMissingPlacementExtension"),
    };

pub const ISSORTED_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ISSORTED_GPU_NONVECTOR_EXTENSION,
    ISSORTED_GPU_MISSING_PLACEMENT_EXTENSION,
];

const ISSORTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
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

pub const ISSORTED_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = issorted(integer_A, integer_dim, direction, options)",
        inputs: &ISSORTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer values are compared exactly and tf is scalar logical. GPU input is documented only for vectors and only with MissingPlacement auto; supported resident forms gather exactly.",
    }];

pub const ISSORTED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSORTED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSORTED_ERRORS,
};
