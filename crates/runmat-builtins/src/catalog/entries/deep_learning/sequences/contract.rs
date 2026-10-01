use crate::*;

const MATRICES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A, B, ...",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more finite numeric scalars or two-dimensional matrices.",
}];
const SEQUENCES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sequences",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of finite numeric sequences.",
};
const DIMENSION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "paddingDim",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive one-based padding dimension.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "PaddingValue, PaddingDirection/Direction, Length, or UniformOutput.",
};
const PAD_INPUTS: [BuiltinParamDescriptor; 3] = [SEQUENCES, DIMENSION, OPTIONS];
const PAD_REQUIRED_INPUTS: [BuiltinParamDescriptor; 2] = [SEQUENCES, DIMENSION];
const MATRIX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double matrix of Cartesian column combinations.",
}];
const PAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host tensor preserving a homogeneous numeric sequence class, or a cell array when UniformOutput is false.",
}];
const PAD_OUTPUT_WITH_MASK: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Padded numeric tensor or cell array.",
    },
    BuiltinParamDescriptor {
        name: "mask",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical tensor or cell array marking original data positions.",
    },
];
const COMBVEC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "M = combvec(A, B, ...)",
    inputs: &MATRICES,
    outputs: &MATRIX,
}];
const PADSEQUENCES_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = padsequences(sequences, paddingDim)",
        inputs: &PAD_REQUIRED_INPUTS,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = padsequences(sequences, paddingDim, Name, Value)",
        inputs: &PAD_INPUTS,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, mask] = padsequences(sequences, paddingDim)",
        inputs: &PAD_REQUIRED_INPUTS,
        outputs: &PAD_OUTPUT_WITH_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, mask] = padsequences(sequences, paddingDim, Name, Value)",
        inputs: &PAD_INPUTS,
        outputs: &PAD_OUTPUT_WITH_MASK,
    },
];

pub const SEQUENCE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "Argument count, numeric data, sequence shape, option name/value, requested output count, or output capacity is invalid.",
    message: "deep learning builtin received invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [SEQUENCE_INVALID_INPUT];

pub const COMBVEC_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "combvec-typed-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "combvec with typed integer inputs is a RunMat extension because the public compatibility reference does not establish those classes",
    error_identifier: Some("RunMat:compatibility:CombvecIntegerInputExtension"),
};
pub const COMBVEC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COMBVEC_INTEGER_EXTENSION];

const COMBVEC_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A, B, ...",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat accepts real typed integer scalars and matrices; values enter the double Cartesian-result domain. MATLAB's public combvec reference does not specify integer input classes.",
}];
const PAD_INTEGER_SEQUENCES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sequences",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "MATLAB documents integer sequence arrays. RunMat preserves exact native storage; mixed-class collections produce double when stacked and retain each class with UniformOutput=false.",
}];
const PAD_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "paddingDim, Length",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Positive dimension and fixed length are decoded exactly from host typed integer storage, with provider-resident scalar fallback where admitted.",
}];
pub const COMBVEC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "combvec(integer_A, integer_B, ...)",
        inputs: &COMBVEC_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The host result is double, so wide integer values can round at output construction; resident numeric inputs gather after metadata preflight.",
    }];
pub const PADSEQUENCES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "padsequences(cell_of_integer_sequences, paddingDim, ...)",
        inputs: &PAD_INTEGER_SEQUENCES,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Homogeneous integer sequences retain their class and wide values through padding and stacking. Mixed-class numeric collections retain RunMat's previous double-result form when stacked; cell output retains each sequence class. The optional mask is logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "padsequences(sequences, integer_paddingDim, Length=integer_length)",
        inputs: &PAD_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Control integers are decoded exactly before use as one-based dimensions or output lengths; numeric output retains a homogeneous sequence class and mask output is logical.",
    },
];
pub const COMBVEC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMBVEC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const PADSEQUENCES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PADSEQUENCES_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
