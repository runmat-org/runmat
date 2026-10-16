use crate::*;

pub const PAGEMTIMES_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pagemtimes-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pagemtimes with typed-integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PagemtimesIntegerInputExtension"),
    };

pub const PAGEMTIMES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [PAGEMTIMES_INTEGER_INPUT_EXTENSION];

pub const PAGEMTIMES_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public X is single or double; RunMat mode admits real integer pages only after an exact binary64-boundary check.",
    }];

pub const PAGEMTIMES_INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public Y is single or double; RunMat mode admits real integer pages only after an exact binary64-boundary check.",
    }];

pub const PAGEMTIMES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = pagemtimes(integer_X, Y, transpose_options...)",
        inputs: &PAGEMTIMES_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer left pages are a gated RunMat extension. Authoritative values cross once into binary64 after exactness checks; a resident result is restored to the owning provider when possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = pagemtimes(X, integer_Y, transpose_options...)",
        inputs: &PAGEMTIMES_INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer right pages are a gated RunMat extension. Authoritative values cross once into binary64 after exactness checks; a resident result is restored to the owning provider when possible.",
    },
];

pub const PAGEMTIMES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Page-wise matrix product.",
}];

pub const PAGEMTIMES_INPUTS_BASIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dense single, double, or complex array.",
    },
];

pub const PAGEMTIMES_INPUTS_TRANSPOSE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "transpX",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`none`, `transpose`, or `ctranspose` for X.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right dense single, double, or complex array.",
    },
    BuiltinParamDescriptor {
        name: "transpY",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`none`, `transpose`, or `ctranspose` for Y.",
    },
];

pub const PAGEMTIMES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Z = pagemtimes(X, Y)",
        inputs: &PAGEMTIMES_INPUTS_BASIC,
        outputs: &PAGEMTIMES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = pagemtimes(X, transpX, Y, transpY)",
        inputs: &PAGEMTIMES_INPUTS_TRANSPOSE,
        outputs: &PAGEMTIMES_OUTPUT,
    },
];

pub const PAGEMTIMES_ERROR_INVALID_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_ARITY",
    identifier: Some("RunMat:pagemtimes:InvalidArity"),
    when: "The call does not use the two-argument or four-argument form.",
    message: "pagemtimes: expected X,Y or X,transpX,Y,transpY",
};

pub const PAGEMTIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_INPUT",
    identifier: Some("RunMat:pagemtimes:InvalidInput"),
    when: "Inputs are not dense single, double, or complex arrays.",
    message: "pagemtimes: inputs must be dense single, double, or complex arrays",
};

pub const PAGEMTIMES_ERROR_INVALID_TRANSPOSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INVALID_TRANSPOSE",
    identifier: Some("RunMat:pagemtimes:InvalidTranspose"),
    when: "Transpose flags are not one of none, transpose, or ctranspose.",
    message: "pagemtimes: transpose options must be 'none', 'transpose', or 'ctranspose'",
};

pub const PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.PAGE_DIM_MISMATCH",
    identifier: Some("RunMat:pagemtimes:PageDimensionMismatch"),
    when: "Trailing page dimensions cannot be implicitly expanded.",
    message: "pagemtimes: page dimensions are not compatible",
};

pub const PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.MATRIX_DIM_MISMATCH",
    identifier: Some("RunMat:pagemtimes:MatrixDimensionMismatch"),
    when: "Matrix dimensions are not valid for page-wise multiplication.",
    message: "pagemtimes: inner matrix dimensions must agree",
};

pub const PAGEMTIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEMTIMES.INTERNAL",
    identifier: Some("RunMat:pagemtimes:InternalError"),
    when: "Runtime cannot materialize the page-wise result.",
    message: "pagemtimes: internal runtime failure",
};

pub const PAGEMTIMES_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PAGEMTIMES_ERROR_INVALID_ARITY,
    PAGEMTIMES_ERROR_INVALID_INPUT,
    PAGEMTIMES_ERROR_INVALID_TRANSPOSE,
    PAGEMTIMES_ERROR_PAGE_DIM_MISMATCH,
    PAGEMTIMES_ERROR_MATRIX_DIM_MISMATCH,
    PAGEMTIMES_ERROR_INTERNAL,
];

pub const PAGEMTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAGEMTIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAGEMTIMES_ERRORS,
};
