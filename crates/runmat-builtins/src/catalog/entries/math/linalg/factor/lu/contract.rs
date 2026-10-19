use crate::*;

pub const LU_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lu-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lu with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LuIntegerInputExtension"),
};
pub const LU_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lu-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lu with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LuLogicalInputExtension"),
};
pub const LU_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [LU_INTEGER_EXTENSION, LU_LOGICAL_EXTENSION];
pub const LU_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode admits integer matrices at an explicit binary64 factorization boundary.",
}];
pub const LU_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[L,U,P] = lu(integer_A)",
        inputs: &LU_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Gated RunMat extension; documented MATLAB input classes remain single and double.",
    }];

pub const LU_OUTPUT_COMBINED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LU",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Combined LU factors.",
}];

pub const LU_OUTPUT_LU: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "L",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower-triangular factor.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper-triangular factor.",
    },
];

pub const LU_OUTPUT_LUP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "L",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower-triangular factor.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper-triangular factor.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation matrix or vector based on pivot mode.",
    },
];

pub const LU_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to factorize.",
}];

pub const LU_INPUTS_A_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "pivotMode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"matrix\""),
        description: "Permutation mode (`\"matrix\"` or `\"vector\"`).",
    },
];

pub const LU_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "LU = lu(A)",
        inputs: &LU_INPUTS_A,
        outputs: &LU_OUTPUT_COMBINED,
    },
    BuiltinSignatureDescriptor {
        label: "LU = lu(A, pivotMode)",
        inputs: &LU_INPUTS_A_MODE,
        outputs: &LU_OUTPUT_COMBINED,
    },
    BuiltinSignatureDescriptor {
        label: "[L, U] = lu(A)",
        inputs: &LU_INPUTS_A,
        outputs: &LU_OUTPUT_LU,
    },
    BuiltinSignatureDescriptor {
        label: "[L, U] = lu(A, pivotMode)",
        inputs: &LU_INPUTS_A_MODE,
        outputs: &LU_OUTPUT_LU,
    },
    BuiltinSignatureDescriptor {
        label: "[L, U, P] = lu(A)",
        inputs: &LU_INPUTS_A,
        outputs: &LU_OUTPUT_LUP,
    },
    BuiltinSignatureDescriptor {
        label: "[L, U, P] = lu(A, pivotMode)",
        inputs: &LU_INPUTS_A_MODE,
        outputs: &LU_OUTPUT_LUP,
    },
];

pub const LU_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LU.INVALID_ARGUMENT",
    identifier: Some("RunMat:lu:InvalidArgument"),
    when: "Option arguments or requested output count are invalid.",
    message: "lu currently supports at most three outputs",
};

pub const LU_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LU.INVALID_INPUT",
    identifier: Some("RunMat:lu:InvalidInput"),
    when: "Input is unsupported for LU factorization.",
    message: "lu: expected numeric or logical input values",
};

pub const LU_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LU.INTERNAL",
    identifier: Some("RunMat:lu:Internal"),
    when: "Runtime cannot materialize LU outputs.",
    message: "lu: internal runtime failure",
};

pub const LU_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LU_ERROR_INVALID_ARGUMENT,
    LU_ERROR_INVALID_INPUT,
    LU_ERROR_INTERNAL,
];

pub const LU_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LU_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LU_ERRORS,
};
