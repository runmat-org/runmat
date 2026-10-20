use crate::*;

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enc",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word encoding compatibility object.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object or word vector.",
}];
const INPUT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object or word vector.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Order, MaxNumWords.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "enc = wordEncoding(documents)",
        inputs: &INPUT,
        outputs: &OUT,
    },
    BuiltinSignatureDescriptor {
        label: "enc = wordEncoding(words)",
        inputs: &INPUT,
        outputs: &OUT,
    },
    BuiltinSignatureDescriptor {
        label: "enc = wordEncoding(documents, Name, Value)",
        inputs: &INPUT_REST,
        outputs: &OUT,
    },
];
pub const WORD_ENCODING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDENCODING.INVALID_INPUT",
    identifier: Some("RunMat:wordEncoding:InvalidInput"),
    when: "Inputs do not match a supported wordEncoding form.",
    message: "wordEncoding received invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [WORD_ENCODING_ERROR_INVALID_INPUT];
pub const WORD_ENCODING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const WORD_ENCODING_INTEGER_MAX_WORDS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "wordencoding-integer-max-num-words",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wordEncoding with a typed-integer MaxNumWords value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WordEncodingIntegerMaxNumWordsExtension"),
    };
pub const WORD_ENCODING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [WORD_ENCODING_INTEGER_MAX_WORDS_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "MaxNumWords", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The public reference specifies a positive integer value or Inf without publishing native integer storage classes. RunMat mode decodes a typed scalar exactly as a bounded vocabulary length." }];
pub const WORD_ENCODING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "enc = wordEncoding(documents, 'MaxNumWords', integer_n)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The exact positive count truncates the ranked host vocabulary and does not enter floating arithmetic. Ordinary positive integer-valued double and positive Inf retain their documented behavior." }];
