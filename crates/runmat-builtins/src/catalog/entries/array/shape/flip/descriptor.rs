use crate::*;

pub const FLIP_TYPED_DIMENSION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "flip-typed-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "flip(A,dim) with a typed-integer or logical dim is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FlipTypedDimensionExtension"),
};
pub const FLIP_DIMENSION_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "flip-dimension-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "flip(A,dims) with multiple or zero dimension entries is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FlipDimensionVectorExtension"),
    };
pub const FLIP_DIRECTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "flip-direction-keyword",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "flip(A,direction) is a RunMat convenience extension",
    error_identifier: Some("RunMat:compatibility:FlipDirectionExtension"),
};
pub const FLIP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FLIP_TYPED_DIMENSION_EXTENSION,
    FLIP_DIMENSION_VECTOR_EXTENSION,
    FLIP_DIRECTION_EXTENSION,
];

const FLIP_INTEGER_A_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are reordered through authoritative storage without numeric conversion.",
    }];
const FLIP_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer dimensions are exact structural controls and are gated as a RunMat extension.",
    }];
pub const FLIP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = flip(integer_A[,dim])",
        inputs: &FLIP_INTEGER_A_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The operation only permutes elements; GPU fallback gathers and re-uploads exact integer storage to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = flip(A,integer_dim)",
        inputs: &FLIP_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict compatibility rejects this extension before parsing the dimension value.",
    },
];

const FLIP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array with selected dimensions reversed.",
}];

const FLIP_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array/value to reverse.",
}];

const FLIP_INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value to reverse.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_direction",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Dimension index/vector or direction keyword ('horizontal', 'vertical', 'both').",
    },
];

const FLIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = flip(A)",
        inputs: &FLIP_INPUTS_A,
        outputs: &FLIP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = flip(A, dim_or_direction)",
        inputs: &FLIP_INPUTS_A_DIM,
        outputs: &FLIP_OUTPUT,
    },
];

pub const FLIP_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.TOO_MANY_INPUTS",
    identifier: Some("RunMat:flip:TooManyInputs"),
    when: "More than one optional argument is provided after A.",
    message: "flip: too many input arguments",
};

pub const FLIP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.INVALID_INPUT",
    identifier: Some("RunMat:flip:InvalidInput"),
    when: "Input type, dimension argument, or direction token is invalid.",
    message: "flip: invalid input argument",
};

pub const FLIP_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:flip:UnsupportedInput"),
    when: "Input type is unsupported for flip.",
    message: "flip: unsupported input type",
};

const FLIP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FLIP_ERROR_TOO_MANY_INPUTS,
    FLIP_ERROR_INVALID_INPUT,
    FLIP_ERROR_UNSUPPORTED_INPUT,
];

pub const FLIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLIP_ERRORS,
};
