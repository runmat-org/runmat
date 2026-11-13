use crate::*;

pub const ISHERMITIAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the matrix satisfies the selected Hermitian predicate.",
}];

pub const ISHERMITIAN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const ISHERMITIAN_INPUTS_OPTION: [BuiltinParamDescriptor; 2] = [
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
        description: "Hermitian mode flag or tolerance scalar.",
    },
];

pub const ISHERMITIAN_INPUTS_FLAG_TOL: [BuiltinParamDescriptor; 3] = [
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
        description: "Mode flag (for example \"hermitian\" or \"skew\").",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tolerance for Hermitian comparison checks.",
    },
];

pub const ISHERMITIAN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "tf = ishermitian(A)",
        inputs: &ISHERMITIAN_INPUTS,
        outputs: &ISHERMITIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ishermitian(A, option)",
        inputs: &ISHERMITIAN_INPUTS_OPTION,
        outputs: &ISHERMITIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ishermitian(A, flag, tol)",
        inputs: &ISHERMITIAN_INPUTS_FLAG_TOL,
        outputs: &ISHERMITIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ishermitian(A, tol, flag)",
        inputs: &ISHERMITIAN_INPUTS_FLAG_TOL,
        outputs: &ISHERMITIAN_OUTPUT,
    },
];

pub const ISHERMITIAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISHERMITIAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:ishermitian:InvalidArgument"),
    when: "Flag/tolerance arguments are malformed, duplicated, or unsupported.",
    message: "ishermitian: invalid argument",
};

pub const ISHERMITIAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISHERMITIAN.INVALID_INPUT",
    identifier: Some("RunMat:ishermitian:InvalidInput"),
    when: "Input shape/type cannot be interpreted as a numeric or logical 2-D matrix.",
    message: "ishermitian: invalid input",
};

pub const ISHERMITIAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISHERMITIAN.INTERNAL",
    identifier: Some("RunMat:ishermitian:Internal"),
    when: "Runtime fails while creating internal tensors or coercing values.",
    message: "ishermitian: internal runtime failure",
};

pub const ISHERMITIAN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ISHERMITIAN_ERROR_INVALID_ARGUMENT,
    ISHERMITIAN_ERROR_INVALID_INPUT,
    ISHERMITIAN_ERROR_INTERNAL,
];

pub const ISHERMITIAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISHERMITIAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISHERMITIAN_ERRORS,
};

pub const ISHERMITIAN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ishermitian-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ishermitian with typed-integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IshermitianIntegerInputExtension"),
    };
pub const ISHERMITIAN_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ishermitian-tolerance-form",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ishermitian tolerance arguments are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IshermitianToleranceExtension"),
    };
pub const ISHERMITIAN_FLAG_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ishermitian-flag-aliases",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ishermitian flag aliases beyond 'nonskew' and 'skew' are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:IshermitianFlagAliasExtension"),
    };
pub const ISHERMITIAN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ISHERMITIAN_INTEGER_INPUT_EXTENSION,
    ISHERMITIAN_TOLERANCE_EXTENSION,
    ISHERMITIAN_FLAG_ALIAS_EXTENSION,
];
pub const ISHERMITIAN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "MATLAB documents single, double, and logical matrix inputs; RunMat integer matrices and tolerance arguments are separately declared and gated extensions.",
    };
