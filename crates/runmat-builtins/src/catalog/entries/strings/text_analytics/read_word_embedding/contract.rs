use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "emb",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word embedding compatibility object.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 word2vec/GloVe text file or zip file containing one.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "emb = readWordEmbedding(filename)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const READ_WORD_EMBEDDING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.READWORDEMBEDDING.INVALID_INPUT",
        identifier: Some("RunMat:readWordEmbedding:InvalidInput"),
        when: "Inputs do not match a supported readWordEmbedding form.",
        message: "readWordEmbedding received invalid input",
    };
pub const READ_WORD_EMBEDDING_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READWORDEMBEDDING.IO",
    identifier: Some("RunMat:readWordEmbedding:IOError"),
    when: "The requested word embedding file cannot be read.",
    message: "Unable to read word embedding file",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    READ_WORD_EMBEDDING_ERROR_INVALID_INPUT,
    READ_WORD_EMBEDDING_ERROR_IO,
];
pub const READ_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const READ_WORD_EMBEDDING_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "readWordEmbedding accepts a host string scalar, character vector, or one-cell character-vector filename and returns a wordEmbedding object. Numeric, logical, symbolic, and provider-resident filename values reject before gather, provider access, or filesystem access.",
    };
