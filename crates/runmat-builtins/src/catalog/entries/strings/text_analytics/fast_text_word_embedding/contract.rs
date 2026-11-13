use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "emb",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word embedding compatibility object.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "emb = fastTextWordEmbedding",
    inputs: &[],
    outputs: &OUTPUTS,
}];
pub const FASTTEXT_WORD_EMBEDDING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FASTTEXTWORDEMBEDDING.INVALID_INPUT",
        identifier: Some("RunMat:fastTextWordEmbedding:InvalidInput"),
        when: "Inputs do not match the supported fastTextWordEmbedding form.",
        message: "fastTextWordEmbedding received invalid input",
    };
const ERRORS: [BuiltinErrorDescriptor; 1] = [FASTTEXT_WORD_EMBEDDING_ERROR_INVALID_INPUT];
pub const FASTTEXT_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
