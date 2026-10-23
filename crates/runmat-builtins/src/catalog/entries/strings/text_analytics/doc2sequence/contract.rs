use crate::*;

pub const DOC2SEQUENCE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOC2SEQUENCE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:doc2sequence:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "doc2sequence: too many output arguments",
};

const OUT_SEQUENCES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sequences",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of document embedding-vector or word-index sequences.",
}];

const IN_MAP_DOCUMENTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
];

const IN_MAP_DOCUMENTS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: UnknownWord, PaddingDirection, PaddingValue, Length.",
    },
];

pub const DOC2SEQUENCE_ERROR_DOC2SEQUENCE_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.DOC2SEQUENCE.INVALID_INPUT",
        identifier: Some("RunMat:doc2sequence:InvalidInput"),
        when: "Inputs do not match a supported doc2sequence form.",
        message: "doc2sequence received invalid input",
    };

const DOC2SEQUENCE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    DOC2SEQUENCE_ERROR_DOC2SEQUENCE_INVALID_INPUT,
    DOC2SEQUENCE_ERROR_TOO_MANY_OUTPUTS,
];

pub const DOC2SEQUENCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(emb, documents)",
            inputs: &IN_MAP_DOCUMENTS,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(enc, documents)",
            inputs: &IN_MAP_DOCUMENTS,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(emb, documents, Name, Value)",
            inputs: &IN_MAP_DOCUMENTS_REST,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(enc, documents, Name, Value)",
            inputs: &IN_MAP_DOCUMENTS_REST,
            outputs: &OUT_SEQUENCES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DOC2SEQUENCE_ERRORS,
};

const DOC2SEQUENCE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Length",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents every built-in integer class and integer-valued single/double scalars; typed storage is validated exactly before conversion to a platform length.",
    },
    BuiltinIntegerInputCapability {
        name: "PaddingValue",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents every built-in integer class for the scalar padding value; it is converted at the floating sequence-output boundary.",
    },
];

pub const DOC2SEQUENCE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "sequences = doc2sequence(enc_or_emb, documents, Length=integer_length, PaddingValue=integer_value)",
        inputs: &DOC2SEQUENCE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Length is an exact positive structural control; PaddingValue crosses explicitly to RunMat's current host floating sequence storage. doc2sequence has no documented GPU-array capability.",
    }];
