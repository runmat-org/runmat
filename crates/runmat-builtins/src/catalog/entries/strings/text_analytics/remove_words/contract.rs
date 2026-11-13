use super::super::model_contract::text_control_gpu_extension;
use crate::*;

pub const REMOVE_WORDS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REMOVE_WORDS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:removeWords:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "removeWords: too many output arguments",
};

const OUT_DOCUMENTS_OR_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocumentsOrBag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument or bagOfWords object.",
}];

const IN_REMOVE_WORDS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "documentsOrBag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or bagOfWords object.",
    },
    BuiltinParamDescriptor {
        name: "wordsOrIdx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to remove or indices into the object's Vocabulary.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

pub const REMOVE_WORDS_ERROR_REMOVE_WORDS_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVEWORDS.INVALID_INPUT",
        identifier: Some("RunMat:removeWords:InvalidInput"),
        when: "Inputs do not match a supported removeWords form.",
        message: "removeWords: invalid input",
    };

const REMOVE_WORDS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    REMOVE_WORDS_ERROR_REMOVE_WORDS_INVALID_INPUT,
    REMOVE_WORDS_ERROR_TOO_MANY_OUTPUTS,
];

pub const REMOVE_WORDS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    text_control_gpu_extension(
        "removewords-explicit-gpu-input",
        "removeWords host fallback for explicit gpuArray input is a RunMat extension",
        "RunMat:compatibility:RemoveWordsExplicitGpuInputExtension",
    );

pub const REMOVE_WORDS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REMOVE_WORDS_EXPLICIT_GPU_EXTENSION];

const REMOVE_WORDS_INDEX_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "idx",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric vocabulary indices accept all eight integer classes and are read exactly before one-based bounds validation and deduplication.",
    }];

pub const REMOVE_WORDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "newDocumentsOrBag = removeWords(documentsOrBag, integer_idx)",
        inputs: &REMOVE_WORDS_INDEX_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact positive indices select vocabulary entries without binary64 collapse; logical masks remain a distinct documented selector form.",
    }];

pub const REMOVE_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(documentsOrBag, words)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(documentsOrBag, idx)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(___, 'IgnoreCase', tf)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_WORDS_ERRORS,
};
