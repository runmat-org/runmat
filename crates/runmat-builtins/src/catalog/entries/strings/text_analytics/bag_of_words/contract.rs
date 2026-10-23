use crate::*;

pub const BAG_OF_WORDS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAG_OF_WORDS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:bagOfWords:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "bagOfWords: too many output arguments",
};

const OUT_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bag-of-words model object.",
}];

const IN_DOCUMENTS_OR_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized documents, word vector, or unique vocabulary.",
}];

const IN_WORDS_COUNTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "uniqueWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique words.",
    },
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word counts per document.",
    },
];

pub const BAG_OF_WORDS_ERROR_BAG_OF_WORDS_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.BAGOFWORDS.INVALID_INPUT",
        identifier: Some("RunMat:bagOfWords:InvalidInput"),
        when: "Inputs do not match a supported bagOfWords form.",
        message: "bagOfWords: invalid input",
    };

const BAG_OF_WORDS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BAG_OF_WORDS_ERROR_BAG_OF_WORDS_INVALID_INPUT,
    BAG_OF_WORDS_ERROR_TOO_MANY_OUTPUTS,
];

pub const BAG_OF_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords",
            inputs: &[],
            outputs: &OUT_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords(documents)",
            inputs: &IN_DOCUMENTS_OR_WORDS,
            outputs: &OUT_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords(uniqueWords, counts)",
            inputs: &IN_WORDS_COUNTS,
            outputs: &OUT_BAG,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BAG_OF_WORDS_ERRORS,
};

const BAG_OF_WORDS_COUNTS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "counts",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Explicit count matrices accept every integer class and require nonnegative values.",
    }];

pub const BAG_OF_WORDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "bag = bagOfWords(uniqueWords, integer_counts)",
        inputs: &BAG_OF_WORDS_COUNTS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Counts remain authoritative through nonnegative-integer validation and vocabulary-column filtering before one conversion into the model's documented double Counts property; the result is an opaque text-model object.",
    }];
