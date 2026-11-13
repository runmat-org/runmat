use crate::*;

const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Paired, layer-specific recognized property names and values; duplicate names are invalid.",
};
const SIZE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "size",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer scalar or, for input layers, vector of positive integers.",
};
const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "layer",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host layer metadata object with the named runtime class.",
}];
const NO_SIZES: [BuiltinParamDescriptor; 1] = [OPTIONS];
const ONE_SIZE: [BuiltinParamDescriptor; 2] = [SIZE, OPTIONS];
const TWO_SIZES: [BuiltinParamDescriptor; 3] = [SIZE, SIZE, OPTIONS];
const FC_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive operation dimension or spatial-channel (R2026a positional form).",
};
const FC_WITH_DIM: [BuiltinParamDescriptor; 3] = [SIZE, FC_DIM, OPTIONS];
const OPTIONAL_ALPHA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "alpha",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Finite activation scale.",
    },
    OPTIONS,
];

pub const LAYER_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "A required layer size, numeric alpha, or name-value option is invalid, or more than one output is requested.",
    message: "deep learning layer received invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [LAYER_INVALID_INPUT];

pub const FEATURE_INPUT_VECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feature-input-vector-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "featureInputLayer accepts a vector of feature dimensions as RunMat metadata; MATLAB documents a scalar feature count",
    error_identifier: Some("RunMat:compatibility:FeatureInputVectorExtension"),
};
pub const FEATURE_INPUT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [FEATURE_INPUT_VECTOR_EXTENSION];

macro_rules! structural_integer_capability {
    ($inputs:ident, $capabilities:ident, $form:literal, $control:literal) => {
        const $inputs: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
            name: $control,
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::Documented,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Host typed integer structural sizes are decoded exactly and admitted only when their stored double-valued property is exact.",
        }];
        pub const $capabilities: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
            form: $form,
            inputs: &$inputs,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::HostOnly,
            overload: BuiltinIntegerOverloadKind::StructuralParameter,
            notes: "The result is a host layer object with double-valued size metadata. Exact wide typed integer sizes are accepted when representable by the host and binary64; inexact values reject rather than round.",
        }];
    };
}

structural_integer_capability!(
    FC_INTEGER_INPUTS,
    FC_INTEGER_CAPABILITIES,
    "fullyConnectedLayer(integer_outputSize, ...)",
    "outputSize"
);
const FEATURE_SCALAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "numFeatures",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Host typed integer scalar feature counts are decoded exactly before storage as double metadata.",
}];
const FEATURE_VECTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "inputSize vector",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat permits an integer vector as feature metadata; MATLAB documents a scalar feature count.",
}];
pub const FEATURE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "featureInputLayer(integer_numFeatures, ...)",
        inputs: &FEATURE_SCALAR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The host layer object stores an exact double InputSize; an inexact typed integer rejects.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "featureInputLayer(integer_inputSize_vector, ...)",
        inputs: &FEATURE_VECTOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This is RunMat-only metadata; the current model forward path rejects vector InputSize rather than ignoring trailing dimensions.",
    },
];
structural_integer_capability!(
    SEQUENCE_INTEGER_INPUTS,
    SEQUENCE_INTEGER_CAPABILITIES,
    "sequenceInputLayer(integer_inputSize, ...)",
    "inputSize"
);
structural_integer_capability!(
    LSTM_INTEGER_INPUTS,
    LSTM_INTEGER_CAPABILITIES,
    "lstmLayer(integer_numHiddenUnits, ...)",
    "numHiddenUnits"
);
structural_integer_capability!(
    BILSTM_INTEGER_INPUTS,
    BILSTM_INTEGER_CAPABILITIES,
    "bilstmLayer(integer_numHiddenUnits, ...)",
    "numHiddenUnits"
);
structural_integer_capability!(
    CONV_INTEGER_INPUTS,
    CONV_INTEGER_CAPABILITIES,
    "convolution1dLayer(integer_filterSize, integer_numFilters, ...)",
    "filterSize, numFilters"
);

macro_rules! layer_descriptor {
    ($signatures:ident, $descriptor:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT,
        }];
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

const FULLY_CONNECTED_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "layer = fullyConnectedLayer(outputSize, Name, Value)",
        inputs: &ONE_SIZE,
        outputs: &OUT,
    },
    BuiltinSignatureDescriptor {
        label: "layer = fullyConnectedLayer(outputSize, dim, Name, Value)",
        inputs: &FC_WITH_DIM,
        outputs: &OUT,
    },
];
pub const FULLY_CONNECTED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FULLY_CONNECTED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
layer_descriptor!(
    FEATURE_INPUT_SIGNATURES,
    FEATURE_INPUT_DESCRIPTOR,
    "layer = featureInputLayer(inputSize, Name, Value)",
    ONE_SIZE
);
layer_descriptor!(
    SEQUENCE_INPUT_SIGNATURES,
    SEQUENCE_INPUT_DESCRIPTOR,
    "layer = sequenceInputLayer(inputSize, Name, Value)",
    ONE_SIZE
);
layer_descriptor!(
    RELU_SIGNATURES,
    RELU_DESCRIPTOR,
    "layer = reluLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    ELU_SIGNATURES,
    ELU_DESCRIPTOR,
    "layer = eluLayer(alpha, Name, Value)",
    OPTIONAL_ALPHA
);
layer_descriptor!(
    SOFTMAX_SIGNATURES,
    SOFTMAX_DESCRIPTOR,
    "layer = softmaxLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    CLASSIFICATION_SIGNATURES,
    CLASSIFICATION_DESCRIPTOR,
    "layer = classificationLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    REGRESSION_SIGNATURES,
    REGRESSION_DESCRIPTOR,
    "layer = regressionLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    LAYER_NORM_SIGNATURES,
    LAYER_NORM_DESCRIPTOR,
    "layer = layerNormalizationLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    GLOBAL_POOL_SIGNATURES,
    GLOBAL_POOL_DESCRIPTOR,
    "layer = globalAveragePooling1dLayer(Name, Value)",
    NO_SIZES
);
layer_descriptor!(
    LSTM_SIGNATURES,
    LSTM_DESCRIPTOR,
    "layer = lstmLayer(numHiddenUnits, Name, Value)",
    ONE_SIZE
);
layer_descriptor!(
    BILSTM_SIGNATURES,
    BILSTM_DESCRIPTOR,
    "layer = bilstmLayer(numHiddenUnits, Name, Value)",
    ONE_SIZE
);
layer_descriptor!(
    CONVOLUTION_SIGNATURES,
    CONVOLUTION_DESCRIPTOR,
    "layer = convolution1dLayer(filterSize, numFilters, Name, Value)",
    TWO_SIZES
);
