use crate::*;

pub const RREF_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reduced row echelon form of A.",
}];

pub const RREF_OUTPUT_R_PIVOTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduced row echelon form of A.",
    },
    BuiltinParamDescriptor {
        name: "pivots",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based pivot column indices.",
    },
];

pub const RREF_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const RREF_INPUTS_A_TOL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Pivot tolerance.",
    },
];

pub const RREF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "R = rref(A)",
        inputs: &RREF_INPUTS_A,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rref(A, tol)",
        inputs: &RREF_INPUTS_A_TOL,
        outputs: &RREF_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivots] = rref(A)",
        inputs: &RREF_INPUTS_A,
        outputs: &RREF_OUTPUT_R_PIVOTS,
    },
    BuiltinSignatureDescriptor {
        label: "[R, pivots] = rref(A, tol)",
        inputs: &RREF_INPUTS_A_TOL,
        outputs: &RREF_OUTPUT_R_PIVOTS,
    },
];

pub const RREF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_ARGUMENT",
    identifier: Some("RunMat:rref:InvalidArgument"),
    when: "Optional tolerance argument is malformed or outside accepted bounds.",
    message: "rref: invalid argument",
};

pub const RREF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INVALID_INPUT",
    identifier: Some("RunMat:rref:InvalidInput"),
    when: "Input shape/type cannot be processed for row reduction.",
    message: "rref: invalid input",
};

pub const RREF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rref:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "rref: too many output arguments",
};

pub const RREF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RREF.INTERNAL",
    identifier: Some("RunMat:rref:Internal"),
    when: "Runtime fails while computing row reduction or executing gather/upload paths.",
    message: "rref: internal runtime failure",
};

pub const RREF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RREF_ERROR_INVALID_ARGUMENT,
    RREF_ERROR_INVALID_INPUT,
    RREF_ERROR_TOO_MANY_OUTPUTS,
    RREF_ERROR_INTERNAL,
];

pub const RREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RREF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RREF_ERRORS,
};

pub const RREF_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rref-integer-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rref accepts typed-integer matrix data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RrefIntegerMatrixExtension"),
};
pub const RREF_INTEGER_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rref-integer-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rref accepts a typed-integer tolerance as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RrefIntegerToleranceExtension"),
    };
pub const RREF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RREF_INTEGER_MATRIX_EXTENSION,
    RREF_INTEGER_TOLERANCE_EXTENSION,
];
pub const RREF_INTEGER_MATRIX_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double matrices; RunMat admits typed integers only when every matrix entry is exactly representable in binary64.",
    }];
pub const RREF_INTEGER_TOLERANCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "tol",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents a single or double tolerance; the RunMat-only typed scalar form is exactness- and range-checked before conversion.",
    }];
pub const RREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[R,pivots] = rref(integer_A [, tol])",
        inputs: &RREF_INTEGER_MATRIX_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The guarded extension crosses once into the double row-reduction algorithm; R and one-based pivot indices are double, and resident fallback restores R through the exact input owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[R,pivots] = rref(A, integer_tol)",
        inputs: &RREF_INTEGER_TOLERANCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The tolerance is a scalar floating-control boundary and never routes an inexact wide integer through binary64.",
    },
];
