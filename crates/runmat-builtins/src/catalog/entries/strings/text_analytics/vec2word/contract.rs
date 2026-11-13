use crate::*;

pub const VEC2WORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VEC2WORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:vec2word:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "vec2word: too many output arguments",
};

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closest vocabulary words.",
}];

const OUT_WORDS_DIST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Closest vocabulary words.",
    },
    BuiltinParamDescriptor {
        name: "dist",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Distances to input vectors.",
    },
];

const IN_VECTORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Embedding vectors, one vector per row.",
    },
];

const IN_VECTORS_REST: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Embedding vectors, one vector per row.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Number of nearest words.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Distance ('cosine' or 'euclidean').",
    },
];

pub const VEC2WORD_ERROR_VEC2WORD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VEC2WORD.INVALID_INPUT",
    identifier: Some("RunMat:vec2word:InvalidInput"),
    when: "Inputs do not match a supported vec2word form.",
    message: "vec2word received invalid input",
};

const VEC2WORD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    VEC2WORD_ERROR_VEC2WORD_INVALID_INPUT,
    VEC2WORD_ERROR_TOO_MANY_OUTPUTS,
];

pub const VEC2WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "words = vec2word(emb, M)",
            inputs: &IN_VECTORS,
            outputs: &OUT_WORDS,
        },
        BuiltinSignatureDescriptor {
            label: "[words, dist] = vec2word(emb, M, k, 'Distance', distance)",
            inputs: &IN_VECTORS_REST,
            outputs: &OUT_WORDS_DIST,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VEC2WORD_ERRORS,
};

pub const VEC2WORD_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "vec2word-integer-vectors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "vec2word with typed-integer embedding vectors is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Vec2wordIntegerVectorsExtension"),
    };

pub const VEC2WORD_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "vec2word-integer-k",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "vec2word with a typed-integer neighbor count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Vec2wordIntegerKExtension"),
};

pub const VEC2WORD_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    VEC2WORD_INTEGER_MATRIX_EXTENSION,
    VEC2WORD_INTEGER_K_EXTENSION,
];

const VEC2WORD_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts typed-integer query vectors only when every value is exactly representable at the binary64 distance boundary.",
    }];

const VEC2WORD_INTEGER_K_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The typed neighbor count is decoded exactly as a positive platform-sized structural value; it does not enter distance arithmetic.",
    }];

pub const VEC2WORD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "words = vec2word(emb, integer_M, ___)",
        inputs: &VEC2WORD_INTEGER_MATRIX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated integer matrix crosses one checked binary64 cosine or Euclidean distance boundary. Words are strings and optional distances are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "words = vec2word(emb, M, integer_k, ___)",
        inputs: &VEC2WORD_INTEGER_K_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The gated typed k selects an exact result count and never enters floating distance computation.",
    },
];
