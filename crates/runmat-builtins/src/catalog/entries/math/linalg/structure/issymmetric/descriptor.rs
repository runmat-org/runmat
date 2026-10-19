use crate::*;

pub const ISSYMMETRIC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the matrix satisfies the selected symmetry predicate.",
}];

pub const ISSYMMETRIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const ISSYMMETRIC_INPUTS_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Symmetry flag or tolerance scalar.",
    },
];

pub const ISSYMMETRIC_INPUTS_FLAG_TOL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Symmetry flag (for example \"skew\" or \"symmetric\").",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tolerance for element-wise symmetry checks.",
    },
];

pub const ISSYMMETRIC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A)",
        inputs: &ISSYMMETRIC_INPUTS,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, option)",
        inputs: &ISSYMMETRIC_INPUTS_OPTION,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, flag, tol)",
        inputs: &ISSYMMETRIC_INPUTS_FLAG_TOL,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issymmetric(A, tol, flag)",
        inputs: &ISSYMMETRIC_INPUTS_FLAG_TOL,
        outputs: &ISSYMMETRIC_OUTPUT,
    },
];

pub const ISSYMMETRIC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INVALID_ARGUMENT",
    identifier: Some("RunMat:issymmetric:InvalidArgument"),
    when: "Flag/tolerance arguments are malformed, duplicated, or unsupported.",
    message: "issymmetric: invalid argument",
};

pub const ISSYMMETRIC_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INVALID_INPUT",
    identifier: Some("RunMat:issymmetric:InvalidInput"),
    when: "Input shape/type cannot be interpreted as a numeric or logical 2-D matrix.",
    message: "issymmetric: invalid input",
};

pub const ISSYMMETRIC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSYMMETRIC.INTERNAL",
    identifier: Some("RunMat:issymmetric:Internal"),
    when: "Runtime fails while creating internal tensors or coercing values.",
    message: "issymmetric: internal runtime failure",
};

pub const ISSYMMETRIC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ISSYMMETRIC_ERROR_INVALID_ARGUMENT,
    ISSYMMETRIC_ERROR_INVALID_INPUT,
    ISSYMMETRIC_ERROR_INTERNAL,
];

pub const ISSYMMETRIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSYMMETRIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSYMMETRIC_ERRORS,
};

pub const ISSYMMETRIC_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issymmetric-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issymmetric with typed-integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IssymmetricIntegerInputExtension"),
    };
pub const ISSYMMETRIC_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issymmetric-tolerance-form",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issymmetric tolerance arguments are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IssymmetricToleranceExtension"),
    };
pub const ISSYMMETRIC_FLAG_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issymmetric-flag-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issymmetric flag aliases beyond 'nonskew' and 'skew' are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:IssymmetricFlagAliasExtension"),
    };
pub const ISSYMMETRIC_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ISSYMMETRIC_INTEGER_INPUT_EXTENSION,
    ISSYMMETRIC_TOLERANCE_EXTENSION,
    ISSYMMETRIC_FLAG_ALIAS_EXTENSION,
];
pub const ISSYMMETRIC_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "MATLAB documents single, double, and logical matrix inputs; RunMat integer matrices and tolerance arguments are separately declared and gated extensions.",
    };
