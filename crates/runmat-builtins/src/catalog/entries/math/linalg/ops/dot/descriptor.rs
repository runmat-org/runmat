use crate::*;

pub const DOT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dot product result.",
}];

pub const DOT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

pub const DOT_INPUTS_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reduction dimension.",
    },
];

pub const DOT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = dot(A, B)",
        inputs: &DOT_INPUTS,
        outputs: &DOT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = dot(A, B, dim)",
        inputs: &DOT_INPUTS_DIM,
        outputs: &DOT_OUTPUT,
    },
];

pub const DOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:dot:InvalidArgument"),
    when: "Argument count or dimension argument is invalid.",
    message: "dot: invalid argument",
};

pub const DOT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOT.INVALID_INPUT",
    identifier: Some("RunMat:dot:InvalidInput"),
    when: "Inputs are unsupported or incompatible.",
    message: "dot: A and B must be the same size.",
};

pub const DOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOT.INTERNAL",
    identifier: Some("RunMat:dot:Internal"),
    when: "Runtime cannot materialize dot outputs.",
    message: "dot: internal runtime failure",
};

pub const DOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DOT_ERROR_INVALID_ARGUMENT,
    DOT_ERROR_INVALID_INPUT,
    DOT_ERROR_INTERNAL,
];

pub const DOT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dot-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "dot with typed-integer data operands is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DotIntegerDataExtension"),
};

pub const DOT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dot-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "dot with logical data operands is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DotLogicalDataExtension"),
};

pub const DOT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [DOT_INTEGER_DATA_EXTENSION, DOT_LOGICAL_DATA_EXTENSION];

pub const DOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_B",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single/double data only. RunMat mode additionally accepts all eight real typed-integer classes.",
    }];

pub const DOT_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive-integer scalar dimension is read exactly from typed scalar storage before platform-bound validation.",
    }];

pub const DOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = dot(integer_A_or_B, B, dim?)",
        inputs: &DOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Mode-gated integer operands are read from authoritative storage and multiplied exactly when both host operands are integral; each exact product then crosses to the floating reduction/output domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = dot(A, B, integer_dim)",
        inputs: &DOT_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All eight scalar integer classes select the reduction dimension through exact structural decoding.",
    },
];

pub const DOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DOT_ERRORS,
};
