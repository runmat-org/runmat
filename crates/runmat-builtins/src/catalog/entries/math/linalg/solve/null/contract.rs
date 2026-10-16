use crate::*;

pub const NULL_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "null-integer-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "null with a native-class integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NullIntegerMatrixExtension"),
};
pub const NULL_LOGICAL_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "null-logical-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "null with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NullLogicalMatrixExtension"),
};
pub const NULL_RESIDENT_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "null-resident-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "null with an explicit gpuArray tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NullResidentToleranceExtension"),
    };

pub const NULL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    NULL_INTEGER_MATRIX_EXTENSION,
    NULL_LOGICAL_MATRIX_EXTENSION,
    NULL_RESIDENT_TOLERANCE_EXTENSION,
];

pub const NULL_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single and double A; RunMat gates integer matrices and checks the binary64 null-space boundary before materialization.",
}];
pub const NULL_INTEGER_TOLERANCE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "tol",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented real numeric scalar tolerance accepts native integer scalars and converts only when exactly representable as binary64.",
    }];

pub const NULL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "Z = null(integer_A, ___)", inputs: &NULL_INTEGER_MATRIX_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer A is a RunMat-only extension; exact host or resident values enter double SVD/RREF computation and return double bases." },
    BuiltinIntegerCapabilityDescriptor { form: "Z = null(A, integer_tol)", inputs: &NULL_INTEGER_TOLERANCE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Integer tolerance values are parsed exactly before the floating rank threshold; resident tolerance controls are a separately gated extension." },
];

pub const NULL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Basis for the null space of A.",
}];

pub const NULL_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const NULL_INPUTS_A_TOL: [BuiltinParamDescriptor; 2] = [
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
        description: "Rank tolerance.",
    },
];

pub const NULL_INPUTS_A_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "`\"rational\"` requests a row-reduction basis; `\"r\"` is a legacy alias.",
    },
];

pub const NULL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Z = null(A)",
        inputs: &NULL_INPUTS_A,
        outputs: &NULL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = null(A, tol)",
        inputs: &NULL_INPUTS_A_TOL,
        outputs: &NULL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Z = null(A, \"rational\")",
        inputs: &NULL_INPUTS_A_OPTION,
        outputs: &NULL_OUTPUT,
    },
];

pub const NULL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INVALID_ARGUMENT",
    identifier: Some("RunMat:null:InvalidArgument"),
    when: "Optional tolerance or basis option is malformed or outside accepted bounds.",
    message: "null: invalid argument",
};

pub const NULL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INVALID_INPUT",
    identifier: Some("RunMat:null:InvalidInput"),
    when: "Input shape/type cannot be processed for null-space evaluation.",
    message: "null: invalid input",
};

pub const NULL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.INTERNAL",
    identifier: Some("RunMat:null:Internal"),
    when: "Runtime fails while computing the null space or executing fallback/upload paths.",
    message: "null: internal runtime failure",
};
pub const NULL_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NULL.PROVIDER_OWNERSHIP_MISMATCH",
    identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
    when: "A resident input has no exact owning provider.",
    message: "null: no acceleration provider owns the input handle",
};

pub const NULL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    NULL_ERROR_INVALID_ARGUMENT,
    NULL_ERROR_INVALID_INPUT,
    NULL_ERROR_INTERNAL,
    NULL_ERROR_PROVIDER_OWNERSHIP,
];

pub const NULL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NULL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NULL_ERRORS,
};
