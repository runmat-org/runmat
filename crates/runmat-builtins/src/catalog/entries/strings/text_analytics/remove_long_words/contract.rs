use super::super::model_contract::{text_control_gpu_extension, text_length_integer_capability};
use crate::*;

pub const REMOVE_LONG_WORDS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVE_LONG_WORDS.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:removeLongWords:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "removeLongWords: too many output arguments",
    };

const OUT_DOCUMENTS_OR_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocumentsOrBag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument or bagOfWords object.",
}];

const IN_REMOVE_LONG: [BuiltinParamDescriptor; 2] = [
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
        description: "Minimum word length to remove.",
    },
];

pub const REMOVE_LONG_WORDS_ERROR_REMOVE_LONG_WORDS_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVELONGWORDS.INVALID_INPUT",
        identifier: Some("RunMat:removeLongWords:InvalidInput"),
        when: "Inputs do not match a supported removeLongWords form.",
        message: "removeLongWords: invalid input",
    };

const REMOVE_LONG_WORDS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    REMOVE_LONG_WORDS_ERROR_REMOVE_LONG_WORDS_INVALID_INPUT,
    REMOVE_LONG_WORDS_ERROR_TOO_MANY_OUTPUTS,
];

pub const REMOVE_LONG_WORDS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    text_control_gpu_extension(
        "removelongwords-explicit-gpu-input",
        "removeLongWords host fallback for explicit gpuArray input is a RunMat extension",
        "RunMat:compatibility:RemoveLongWordsExplicitGpuInputExtension",
    );

pub const REMOVE_LONG_WORDS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REMOVE_LONG_WORDS_EXPLICIT_GPU_EXTENSION];

pub const REMOVE_LONG_WORDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [text_length_integer_capability(
        "newDocumentsOrBag = removeLongWords(documentsOrBag, integer_len)",
    )];

pub const REMOVE_LONG_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newDocumentsOrBag = removeLongWords(documentsOrBag, len)",
        inputs: &IN_REMOVE_LONG,
        outputs: &OUT_DOCUMENTS_OR_BAG,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_LONG_WORDS_ERRORS,
};
