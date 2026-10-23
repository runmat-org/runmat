use crate::*;
mod controls;
pub use controls::*;

const OUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "wc",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the WordCloudChart object.",
}];

const IN_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text, tokenizedDocument, bag model, categorical array, or LDA model.",
}];

const IN_WORDS_SIZES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word vector.",
    },
    BuiltinParamDescriptor {
        name: "sizeData",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word size vector.",
    },
];

const IN_TABLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "tbl",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table with word and size variables.",
    },
    BuiltinParamDescriptor {
        name: "wordVar",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table variable containing words.",
    },
    BuiltinParamDescriptor {
        name: "sizeVar",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table variable containing word sizes.",
    },
];

const IN_REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "WordCloudChart name-value properties.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(str)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(documents)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(bag)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(words, sizeData)",
        inputs: &IN_WORDS_SIZES,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(tbl, wordVar, sizeVar)",
        inputs: &IN_TABLE,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(C)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(parent, ...)",
        inputs: &IN_REST,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(..., Name, Value)",
        inputs: &IN_REST,
        outputs: &OUT_HANDLE,
    },
];

pub const WORDCLOUD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDCLOUD.INVALID_ARGUMENT",
    identifier: Some("RunMat:wordcloud:InvalidArgument"),
    when: "Input data, table selectors, parent axes, or WordCloudChart properties are invalid.",
    message: "wordcloud: invalid argument",
};

pub const WORDCLOUD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDCLOUD.INTERNAL",
    identifier: Some("RunMat:wordcloud:Internal"),
    when: "Internal plotting state update fails while creating or mutating a word cloud.",
    message: "wordcloud: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    WORDCLOUD_ERROR_INVALID_ARGUMENT,
    WORDCLOUD_ERROR_INTERNAL,
    WORDCLOUD_ERROR_TOO_MANY_OUTPUTS,
];

pub const WORDCLOUD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const WORDCLOUD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDCLOUD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:wordcloud:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "wordcloud: too many output arguments",
};
