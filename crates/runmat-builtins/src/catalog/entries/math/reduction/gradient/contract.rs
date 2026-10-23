use crate::*;

const GRADIENT_OUTPUT_G: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Primary gradient component.",
}];

const GRADIENT_OUTPUT_GS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Gi",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Gradient components ordered by MATLAB axis semantics.",
}];

const GRADIENT_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
}];

const GRADIENT_INPUTS_F_H: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Scalar spacing shared across all output dimensions, or a coordinate vector for vector inputs.",
    },
];

const GRADIENT_INPUTS_F_HS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "h_i",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Per-dimension scalar or coordinate-vector spacings (one per gradient dimension).",
    },
];

const GRADIENT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "G = gradient(F)",
        inputs: &GRADIENT_INPUTS_F,
        outputs: &GRADIENT_OUTPUT_G,
    },
    BuiltinSignatureDescriptor {
        label: "G = gradient(F, h)",
        inputs: &GRADIENT_INPUTS_F_H,
        outputs: &GRADIENT_OUTPUT_G,
    },
    BuiltinSignatureDescriptor {
        label: "[G1, G2, ...] = gradient(F)",
        inputs: &GRADIENT_INPUTS_F,
        outputs: &GRADIENT_OUTPUT_GS,
    },
    BuiltinSignatureDescriptor {
        label: "[G1, G2, ...] = gradient(F, h1, h2, ...)",
        inputs: &GRADIENT_INPUTS_F_HS,
        outputs: &GRADIENT_OUTPUT_GS,
    },
];

pub const GRADIENT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INVALID_ARGUMENT",
    identifier: Some("RunMat:gradient:InvalidArgument"),
    when: "Output-count or spacing argument grammar is invalid.",
    message: "gradient: invalid argument",
};

pub const GRADIENT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INVALID_INPUT",
    identifier: Some("RunMat:gradient:InvalidInput"),
    when: "Input value cannot be converted to a supported gradient domain.",
    message: "gradient: invalid input",
};

pub const GRADIENT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INTERNAL",
    identifier: Some("RunMat:gradient:Internal"),
    when: "Gradient execution fails due to gather, conversion, allocation, or indexing operations.",
    message: "gradient: internal failure",
};

pub const GRADIENT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GRADIENT_ERROR_INVALID_ARGUMENT,
    GRADIENT_ERROR_INVALID_INPUT,
    GRADIENT_ERROR_INTERNAL,
];

pub const GRADIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRADIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRADIENT_ERRORS,
};

pub const GRADIENT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gradient-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gradient with typed-integer data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GradientIntegerDataExtension"),
    };

pub const GRADIENT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gradient-logical-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gradient with logical data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GradientLogicalDataExtension"),
    };

pub const GRADIENT_INTEGER_SPACING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gradient-integer-spacing",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gradient with typed-integer spacing is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GradientIntegerSpacingExtension"),
    };

pub const GRADIENT_LOGICAL_SPACING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gradient-logical-spacing",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gradient with logical spacing is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GradientLogicalSpacingExtension"),
    };

pub const GRADIENT_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    GRADIENT_INTEGER_DATA_EXTENSION,
    GRADIENT_LOGICAL_DATA_EXTENSION,
    GRADIENT_INTEGER_SPACING_EXTENSION,
    GRADIENT_LOGICAL_SPACING_EXTENSION,
];

const GRADIENT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "F",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes must be exactly representable at the binary64 finite-difference boundary.",
    }];

const GRADIENT_INTEGER_SPACING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "h or h_i",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar and coordinate-vector integer spacing must be exactly representable at the binary64 finite-difference boundary.",
    }];

pub const GRADIENT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gradient(integer_F, ...)",
        inputs: &GRADIENT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer data crosses one checked binary64 boundary and produces ordinary double gradient components; resident integer buffers never enter floating provider kernels.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gradient(F, integer_h_or_h_i, ...)",
        inputs: &GRADIENT_INTEGER_SPACING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer spacing crosses one checked binary64 boundary; output precision continues to follow the ordinary floating data path.",
    },
];
