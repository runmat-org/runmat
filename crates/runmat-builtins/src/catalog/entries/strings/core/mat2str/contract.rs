use crate::*;

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const MAT2STR_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or logical matrix to serialize.",
}];
const MAT2STR_INPUT_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive number of significant digits.",
    },
];
const MAT2STR_INPUT_CLASS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional positive number of significant digits.",
    },
    BuiltinParamDescriptor {
        name: "class",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"class\""),
        description: "Include the input class constructor in the expression.",
    },
];
const MAT2STR_INPUT_CLASS_ONLY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "class",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"class\""),
        description: "Include the input class constructor in the expression.",
    },
];
const MAT2STR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A)",
        inputs: &MAT2STR_INPUT_A,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, n)",
        inputs: &MAT2STR_INPUT_N,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, 'class')",
        inputs: &MAT2STR_INPUT_CLASS_ONLY,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, n, 'class')",
        inputs: &MAT2STR_INPUT_CLASS,
        outputs: &OUT_ANY,
    },
];
pub const MAT2STR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAT2STR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mat2str:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "mat2str: too many output arguments",
};

pub const MAT2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAT2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MAT2STR_ERROR_TOO_MANY_OUTPUTS],
};

pub const MAT2STR_INTEGER_PRECISION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mat2str-integer-precision",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed integer mat2str precision is a RunMat extension because the public MATLAB page requires a positive integer without enumerating native integer storage classes",
        error_identifier: Some("RunMat:compatibility:Mat2strIntegerPrecisionExtension"),
    };
pub const MAT2STR_TEXT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mat2str-text-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mat2str input outside the documented numeric and logical matrix domain is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Mat2strTextInputExtension"),
};
pub const MAT2STR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    MAT2STR_INTEGER_PRECISION_EXTENSION,
    MAT2STR_TEXT_INPUT_EXTENSION,
];

const MAT2STR_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer matrix classes serialize from exact native values; the 'class' option emits the matching constructor.",
    }];
const MAT2STR_INTEGER_PRECISION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat accepts an exact positive typed integer precision under its extension policy; strict compatibility retains the publicly evidenced floating scalar form.",
    }];
pub const MAT2STR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = mat2str(integer_A[, n][, 'class'])",
        inputs: &MAT2STR_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Decimal serialization never crosses binary64; automatic residency gathers authoritatively, while explicit gpuArray input is unsupported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = mat2str(A, integer_n[, 'class'])",
        inputs: &MAT2STR_INTEGER_PRECISION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The RunMat-only typed precision is read exactly and must be positive; explicit resident precision rejects before transfer.",
    },
];
