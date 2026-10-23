use super::super::model_contract::{text_control_gpu_extension, text_length_integer_capability};
use crate::*;

pub const REMOVE_SHORT_WORDS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVE_SHORT_WORDS.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:removeShortWords:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "removeShortWords: too many output arguments",
    };

const OUT_DOCUMENTS_OR_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocumentsOrBag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument or bagOfWords object.",
}];

const IN_REMOVE_SHORT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrBag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or bagOfWords object.",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum word length to remove.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS_DOCUMENTS.INVALID_INPUT",
    identifier: Some("RunMat:textAnalyticsDocuments:InvalidInput"),
    when: "Inputs do not match a supported Text Analytics document or model helper form.",
    message: "Text Analytics document helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    ERROR_INVALID_INPUT,
    REMOVE_SHORT_WORDS_ERROR_TOO_MANY_OUTPUTS,
];

pub const REMOVE_SHORT_WORDS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    text_control_gpu_extension(
        "removeshortwords-explicit-gpu-input",
        "removeShortWords host fallback for explicit gpuArray input is a RunMat extension",
        "RunMat:compatibility:RemoveShortWordsExplicitGpuInputExtension",
    );

pub const REMOVE_SHORT_WORDS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REMOVE_SHORT_WORDS_EXPLICIT_GPU_EXTENSION];

pub const REMOVE_SHORT_WORDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [text_length_integer_capability(
        "newDocumentsOrBag = removeShortWords(documentsOrBag, integer_len)",
    )];

pub const REMOVE_SHORT_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newDocumentsOrBag = removeShortWords(documentsOrBag, len)",
        inputs: &IN_REMOVE_SHORT,
        outputs: &OUT_DOCUMENTS_OR_BAG,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
