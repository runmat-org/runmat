use crate::*;

pub const IND2WORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2WORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ind2word:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "ind2word: too many output arguments",
};

pub const IND2WORD_TYPED_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ind2word-typed-integer-indices",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ind2word with a typed-integer index vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Ind2wordTypedIntegerExtension"),
    };

pub const IND2WORD_NONVECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ind2word-nonvector-indices",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ind2word with matrix or multidimensional indices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ind2wordNonvectorExtension"),
};

pub const IND2WORD_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ind2word-resident-indices",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ind2word with resident indices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ind2wordResidentExtension"),
};

pub const IND2WORD_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    IND2WORD_TYPED_INTEGER_EXTENSION,
    IND2WORD_NONVECTOR_EXTENSION,
    IND2WORD_RESIDENT_EXTENSION,
];

const IND2WORD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The current public reference specifies positive integer values but does not publish a native numeric class table; typed-integer vectors are therefore gated and read from authoritative integer storage.",
    }];

pub const IND2WORD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "words = ind2word(enc, integer_M)",
        inputs: &IND2WORD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The public form uses a host positive-integer vector and returns a string vector. RunMat gates typed-integer and resident forms independently, reads typed indices exactly from their native class, and always returns host strings.",
    }];

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Words mapped from encoding indices.",
}];

const IN_INDICES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer word encoding indices.",
    },
];

pub const IND2WORD_ERROR_IND2WORD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2WORD.INVALID_INPUT",
    identifier: Some("RunMat:ind2word:InvalidInput"),
    when: "Inputs do not match a supported ind2word form.",
    message: "ind2word received invalid input",
};

const IND2WORD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    IND2WORD_ERROR_IND2WORD_INVALID_INPUT,
    IND2WORD_ERROR_TOO_MANY_OUTPUTS,
];

pub const IND2WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "words = ind2word(enc, M)",
        inputs: &IN_INDICES,
        outputs: &OUT_WORDS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2WORD_ERRORS,
};
