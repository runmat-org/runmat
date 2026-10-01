use crate::*;

pub const TSNE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tsne-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tsne with native-class integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TsneIntegerDataExtension"),
};
pub const TSNE_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tsne-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tsne with native-class integer numeric option values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TsneIntegerOptionExtension"),
};
pub const TSNE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tsne-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tsne host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TsneResidentInputExtension"),
};
pub const TSNE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TSNE_INTEGER_DATA_EXTENSION,
    TSNE_INTEGER_OPTION_EXTENSION,
    TSNE_RESIDENT_INPUT_EXTENSION,
];

const TSNE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer observations are gated before gather and must be exactly representable at the binary64 optimization boundary.",
    }];
const TSNE_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric option value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer InitialY and numeric controls are independently gated and checked before conversion.",
    }];
pub const TSNE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tsne(integer_X, ___)",
        inputs: &TSNE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The embedding and loss remain floating-point results; lossy integer observation conversion rejects.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tsne(X, Name, integer_Value)",
        inputs: &TSNE_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer-valued structural controls are decoded exactly; numeric optimization values cross one checked binary64 boundary.",
    },
];

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Low-dimensional embedding matrix.",
}];

const OUTPUT_Y_LOSS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Low-dimensional embedding matrix.",
    },
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Kullback-Leibler divergence for the final embedding.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Algorithm, Distance, NumDimensions, NumPCAComponents, Perplexity, Standardize, InitialY, LearnRate, Options, Theta, and Verbose.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];

const TSNE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = tsne(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tsne(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, loss] = tsne(___)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_Y_LOSS,
    },
];

pub const TSNE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TSNE.INVALID_ARGUMENT",
    identifier: Some("RunMat:tsne:InvalidArgument"),
    when: "Inputs, dimensions, distances, initialization, or name-value options are malformed or unsupported.",
    message: "tsne: invalid argument",
};

pub const TSNE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TSNE.INTERNAL",
    identifier: Some("RunMat:tsne:Internal"),
    when: "RunMat cannot allocate or construct the t-SNE result.",
    message: "tsne: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [TSNE_ERROR_INVALID_ARGUMENT, TSNE_ERROR_INTERNAL];

pub const TSNE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TSNE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
