use crate::*;

const OUT_NONE: [BuiltinParamDescriptor; 0] = [];

const IN_EMBEDDING_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target UTF-8 word2vec text file.",
    },
];

pub const WRITE_WORD_EMBEDDING_ERROR_WRITE_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.WRITEWORDEMBEDDING.INVALID_INPUT",
        identifier: Some("RunMat:writeWordEmbedding:InvalidInput"),
        when: "Inputs do not match the supported writeWordEmbedding form.",
        message: "writeWordEmbedding received invalid input",
    };

pub const WRITE_WORD_EMBEDDING_ERROR_WRITE_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITEWORDEMBEDDING.IO",
    identifier: Some("RunMat:writeWordEmbedding:IOError"),
    when: "The requested word embedding file cannot be written.",
    message: "Unable to write word embedding file",
};

pub const WRITE_WORD_EMBEDDING_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.WRITEWORDEMBEDDING.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:writeWordEmbedding:TooManyOutputs"),
        when: "An output is requested from this no-output writer.",
        message: "writeWordEmbedding: too many output arguments",
    };

const WRITE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    WRITE_WORD_EMBEDDING_ERROR_WRITE_INVALID_INPUT,
    WRITE_WORD_EMBEDDING_ERROR_WRITE_IO,
    WRITE_WORD_EMBEDDING_ERROR_TOO_MANY_OUTPUTS,
];

pub const WRITE_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "writeWordEmbedding(emb, filename)",
        inputs: &IN_EMBEDDING_FILENAME,
        outputs: &OUT_NONE,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITE_ERRORS,
};

pub const WRITE_WORD_EMBEDDING_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "writeWordEmbedding accepts a wordEmbedding object and a host text filename. Numeric and resident filename values are invalid and reject before provider or filesystem access; numeric vector properties inside the embedding object are model data, not public numeric inputs.",
    };
