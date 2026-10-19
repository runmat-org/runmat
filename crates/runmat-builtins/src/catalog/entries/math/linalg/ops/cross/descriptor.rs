use crate::*;

pub const CROSS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross product result.",
}];

pub const CROSS_INPUTS: [BuiltinParamDescriptor; 2] = [
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

pub const CROSS_INPUTS_DIM: [BuiltinParamDescriptor; 3] = [
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
        description: "Dimension with extent 3.",
    },
];

pub const CROSS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = cross(A, B)",
        inputs: &CROSS_INPUTS,
        outputs: &CROSS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cross(A, B, dim)",
        inputs: &CROSS_INPUTS_DIM,
        outputs: &CROSS_OUTPUT,
    },
];

pub const CROSS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INVALID_ARGUMENT",
    identifier: Some("RunMat:cross:InvalidArgument"),
    when: "Argument count or dimension argument is invalid.",
    message: "cross: invalid argument",
};

pub const CROSS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INVALID_INPUT",
    identifier: Some("RunMat:cross:InvalidInput"),
    when: "Inputs are unsupported or incompatible for cross product.",
    message: "cross: A and B must be the same size.",
};

pub const CROSS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INTERNAL",
    identifier: Some("RunMat:cross:Internal"),
    when: "Runtime cannot materialize cross outputs.",
    message: "cross: internal runtime failure",
};

pub const CROSS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CROSS_ERROR_INVALID_ARGUMENT,
    CROSS_ERROR_INVALID_INPUT,
    CROSS_ERROR_INTERNAL,
];

pub const CROSS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CROSS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CROSS_ERRORS,
};

pub const CROSS_INTEGER_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-a",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer A operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerAExtension"),
};
pub const CROSS_INTEGER_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-b",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer B operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerBExtension"),
};
pub const CROSS_LOGICAL_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-a",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical A operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalAExtension"),
};
pub const CROSS_LOGICAL_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-b",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical B operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalBExtension"),
};
pub const CROSS_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer dim argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerDimExtension"),
};
pub const CROSS_LOGICAL_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical dim argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalDimExtension"),
};
pub const CROSS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    CROSS_INTEGER_A_EXTENSION,
    CROSS_INTEGER_B_EXTENSION,
    CROSS_LOGICAL_A_EXTENSION,
    CROSS_LOGICAL_B_EXTENSION,
    CROSS_INTEGER_DIM_EXTENSION,
    CROSS_LOGICAL_DIM_EXTENSION,
];
pub const CROSS_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "MATLAB documents only single and double data operands; RunMat mode admits every real or complex integer class at an explicit floating boundary, with single output when the other data operand is single and double otherwise." }];
pub const CROSS_INTEGER_B_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "B", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "The independently gated B operand accepts every real or complex integer class in RunMat mode." }];
pub const CROSS_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "dim", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The public cross reference documents a positive-integer scalar dim but does not list typed integer classes. RunMat mode accepts every typed integer class exactly; an ordinary double-valued integer scalar remains documented behavior." }];
pub const CROSS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(integer_A, B, dim?)", inputs: &CROSS_INTEGER_A_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only integer data crosses a floating boundary. The result is single when either data operand is single and double otherwise; matching shapes and the length-three operating dimension are preserved. Resident integer inputs gather exactly before conversion and eligible results return to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(A, integer_B, dim?)", inputs: &CROSS_INTEGER_B_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The B role has the same independently gated RunMat-only floating-boundary behavior and single-dominant output-class rule." },
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(A, B, integer_dim)", inputs: &CROSS_INTEGER_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "dim controls traversal only; floating data output preserves the supported input precision and residency path." },
];
