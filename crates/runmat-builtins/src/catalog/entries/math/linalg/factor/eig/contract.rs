use crate::*;

pub const EIG_NONFLOATING_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eig-nonfloating-coefficient",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eig with an integer or logical coefficient matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EigNonfloatingCoefficientExtension"),
    };

pub const EIG_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [EIG_NONFLOATING_COEFFICIENT_EXTENSION];

pub const EIG_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Admitted integer coefficients must be exactly representable at the explicit binary64 eigensolver boundary.",
}];
pub const EIG_INTEGER_AB_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "First generalized coefficient matrix.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Second generalized coefficient matrix.",
    },
];

pub const EIG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "eig(A, ...) with typed-integer A",
        inputs: &EIG_INTEGER_A_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only coefficient extension with double outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eig(A, B, ...) with a typed-integer coefficient",
        inputs: &EIG_INTEGER_AB_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each integer coefficient is checked independently before binary64 materialization.",
    },
];

pub const EIG_OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Eigenvalues as a column vector.",
}];

pub const EIG_OUTPUT_VD: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix (or vector when `vector` option is used).",
    },
];

pub const EIG_OUTPUT_VDW: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix (or vector when `vector` option is used).",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left eigenvectors.",
    },
];

pub const EIG_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input square matrix.",
}];

pub const EIG_INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional selectors (`balance`, `nobalance`, `vector`, `matrix`).",
    },
];

pub const EIG_INPUTS_AB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second square coefficient of A - lambda*B; singular B and infinite or indeterminate generalized eigenvalues are supported.",
    },
];

pub const EIG_INPUTS_AB_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second square coefficient of A - lambda*B; singular B and infinite or indeterminate generalized eigenvalues are supported.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional selectors (`balance`, `nobalance`, `vector`, `matrix`).",
    },
];

pub const EIG_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "d = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_VDW,
    },
];

pub const EIG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INVALID_ARGUMENT",
    identifier: Some("RunMat:eig:InvalidArgument"),
    when: "Option arguments or requested output count are invalid.",
    message: "eig currently supports at most three outputs",
};

pub const EIG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INVALID_INPUT",
    identifier: Some("RunMat:eig:InvalidInput"),
    when: "Input is unsupported or matrix shape is invalid.",
    message: "eig: input matrix must be square",
};

pub const EIG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INTERNAL",
    identifier: Some("RunMat:eig:Internal"),
    when: "Runtime cannot compute or materialize eig outputs.",
    message: "eig: internal runtime failure",
};

pub const EIG_ERRORS: [BuiltinErrorDescriptor; 3] = [
    EIG_ERROR_INVALID_ARGUMENT,
    EIG_ERROR_INVALID_INPUT,
    EIG_ERROR_INTERNAL,
];

pub const EIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EIG_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EIG_ERRORS,
};
