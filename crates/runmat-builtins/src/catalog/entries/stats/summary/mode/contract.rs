use crate::*;

const MODE_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Most frequent value along the selected dimension.",
}];

const MODE_OUTPUT_MF: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Most frequent value along the selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency counts for each reported mode.",
    },
];

const MODE_OUTPUT_MFC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Most frequent value along the selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequency counts for each reported mode.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell array containing all tied modal values per slice.",
    },
];

const MODE_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data array.",
}];

const MODE_INPUTS_X_DIM_OR_ALL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data array.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_vecdim_or_all",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduction axis, vector of axes, or 'all'.",
    },
];

const MODE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "M = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F] = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_MF,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X)",
        inputs: &MODE_INPUTS_X,
        outputs: &MODE_OUTPUT_MFC,
    },
    BuiltinSignatureDescriptor {
        label: "[M, F, C] = mode(X, dim_or_vecdim_or_all)",
        inputs: &MODE_INPUTS_X_DIM_OR_ALL,
        outputs: &MODE_OUTPUT_MFC,
    },
];

pub const MODE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INVALID_ARGUMENT",
    identifier: Some("RunMat:mode:InvalidArgument"),
    when: "Arguments are malformed, duplicated, or unrecognised.",
    message: "mode: invalid argument",
};

pub const MODE_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INVALID_DIMENSION",
    identifier: Some("RunMat:mode:InvalidDimension"),
    when: "Dimension argument is zero or negative.",
    message: "mode: dimension must be >= 1",
};

pub const MODE_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:mode:ComplexUnsupported"),
    when: "Input data is complex-valued.",
    message: "mode: complex inputs are not supported; gather real data first",
};

pub const MODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MODE.INTERNAL",
    identifier: Some("RunMat:mode:Internal"),
    when: "Internal conversion/allocation/shape handling fails.",
    message: "mode: internal operation failed",
};

const MODE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    MODE_ERROR_INVALID_ARGUMENT,
    MODE_ERROR_INVALID_DIMENSION,
    MODE_ERROR_COMPLEX_UNSUPPORTED,
    MODE_ERROR_INTERNAL,
];

pub const MODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MODE_ERRORS,
};

const MODE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric arrays accept every real integer class; modal values and every tied-value cell preserve the exact input class.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive integer dimensions are parsed exactly from typed integer or integer-valued floating controls.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[M, F, C] = mode(A, dim_or_vecdim_or_all)",
        inputs: &MODE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "M and the sorted tied vectors in C preserve exact same-class integer storage, F is always double, first/smallest tie rules are exact above flintmax, and resident integer inputs gather then re-upload typed value outputs.",
    }];
