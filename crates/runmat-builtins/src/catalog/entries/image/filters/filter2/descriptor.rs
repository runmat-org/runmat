use crate::*;

pub const FILTER2_MODE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filter2-convolution-mode",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filter2 'conv' and 'corr' mode flags are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:Filter2ModeExtension"),
};

pub const FILTER2_ND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filter2-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filter2 inputs with more than two dimensions are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Filter2NdExtension"),
};

pub const FILTER2_VARIADIC_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filter2-variadic-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filter2 with more than one trailing option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Filter2VariadicOptionsExtension"),
    };

pub const FILTER2_INTEGER_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filter2-integer-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filter2 with integer gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Filter2IntegerGpuExtension"),
};

pub const FILTER2_LOGICAL_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filter2-logical-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filter2 with logical gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Filter2LogicalGpuExtension"),
};

pub const FILTER2_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    FILTER2_MODE_EXTENSION,
    FILTER2_ND_EXTENSION,
    FILTER2_VARIADIC_OPTIONS_EXTENSION,
    FILTER2_INTEGER_GPU_EXTENSION,
    FILTER2_LOGICAL_GPU_EXTENSION,
];

const FILTER2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "H",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for the coefficient matrix.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for the input matrix.",
    },
];

pub const FILTER2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = filter2(integer_H, integer_X, shape)",
        inputs: &FILTER2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer matrices enter the documented double computation domain; MATLAB GPU inputs are restricted to single and double.",
    }];

const FILTER2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered output image/array.",
}];

const FILTER2_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
];

const FILTER2_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional filter2 flags: 'same'|'full'|'valid'|'conv'|'corr'.",
    },
];

const FILTER2_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = filter2(h, X)",
        inputs: &FILTER2_INPUTS_CORE,
        outputs: &FILTER2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = filter2(h, X, options...)",
        inputs: &FILTER2_INPUTS_WITH_OPTIONS,
        outputs: &FILTER2_OUTPUT,
    },
];

pub const FILTER2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER2.INVALID_INPUT",
    identifier: Some("RunMat:filter2:InvalidInput"),
    when: "Input/kernel tensors are invalid or cannot be converted.",
    message: "filter2: invalid input",
};

pub const FILTER2_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER2.INVALID_OPTION",
    identifier: Some("RunMat:filter2:InvalidOption"),
    when: "One or more option flags are invalid.",
    message: "filter2: invalid option",
};

pub const FILTER2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER2.INTERNAL",
    identifier: Some("RunMat:filter2:Internal"),
    when: "Internal filtering operation fails.",
    message: "filter2: internal operation failed",
};

const FILTER2_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FILTER2_ERROR_INVALID_INPUT,
    FILTER2_ERROR_INVALID_OPTION,
    FILTER2_ERROR_INTERNAL,
];

pub const FILTER2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILTER2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILTER2_ERRORS,
};
