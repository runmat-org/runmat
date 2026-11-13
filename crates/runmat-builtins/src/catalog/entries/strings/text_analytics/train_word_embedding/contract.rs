use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "emb",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word embedding compatibility object.",
}];
const SOURCE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "source",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 text filename or tokenizedDocument object.",
}];
const SOURCE_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "source",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "UTF-8 text filename or tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options controlling local deterministic embedding training.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "emb = trainWordEmbedding(filename)",
        inputs: &SOURCE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "emb = trainWordEmbedding(documents)",
        inputs: &SOURCE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "emb = trainWordEmbedding(___, Name, Value)",
        inputs: &SOURCE_REST,
        outputs: &OUTPUTS,
    },
];
pub const TRAIN_WORD_EMBEDDING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.TRAINWORDEMBEDDING.INVALID_INPUT",
        identifier: Some("RunMat:trainWordEmbedding:InvalidInput"),
        when: "Inputs do not match a supported trainWordEmbedding form.",
        message: "trainWordEmbedding received invalid input",
    };
const ERRORS: [BuiltinErrorDescriptor; 1] = [TRAIN_WORD_EMBEDDING_ERROR_INVALID_INPUT];
pub const TRAIN_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const TRAIN_WORD_EMBEDDING_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "train-word-embedding-native-integer-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "trainWordEmbedding accepts native integer option values",
        error_identifier: Some("RunMat:compatibility:TrainWordEmbeddingIntegerControlExtension"),
    };
pub const TRAIN_WORD_EMBEDDING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TRAIN_WORD_EMBEDDING_INTEGER_CONTROL_EXTENSION];

const INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 10] = [
    integer_control("Dimension"),
    integer_control("Window"),
    integer_control("DiscardFactor"),
    integer_control("NumNegativeSamples"),
    integer_control("NumEpochs"),
    integer_control("MinCount"),
    integer_control("NGramRange"),
    integer_control("InitialLearnRate"),
    integer_control("UpdateRate"),
    integer_control("Verbose"),
];
const fn integer_control(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public option is integer-valued numeric data without a documented native-integer class surface; native integer storage is admitted only in RunMat mode and validated exactly for its role.",
    }
}
pub const TRAIN_WORD_EMBEDDING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "emb = trainWordEmbedding(source, Name, native_integer_value, ...)",
        inputs: &INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Counts, sizes, ranges, and boolean controls decode exactly in every supported integer class. Floating training-rate options require exact binary64 representation. Automatic residency gathers transparently; explicit gpuArray option intent is unsupported.",
    }];
