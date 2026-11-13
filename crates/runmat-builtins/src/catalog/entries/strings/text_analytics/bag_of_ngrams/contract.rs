use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bag-of-n-grams model object.",
}];

const DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized documents or a single-document word vector.",
}];

const DOCUMENTS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Tokenized documents or a single-document word vector.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option: NgramLengths.",
    },
];

const NGRAMS_COUNTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "uniqueNgrams",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique n-gram string matrix.",
    },
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-gram counts per document.",
    },
];

const NGRAMS_COUNTS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "uniqueNgrams",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique n-gram string matrix.",
    },
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-gram counts per document.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option: NgramLengths.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "bag = bagOfNgrams",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "bag = bagOfNgrams(documents)",
        inputs: &DOCUMENTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "bag = bagOfNgrams(___, 'NgramLengths', lengths)",
        inputs: &DOCUMENTS_REST,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "bag = bagOfNgrams(uniqueNgrams, counts)",
        inputs: &NGRAMS_COUNTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "bag = bagOfNgrams(uniqueNgrams, counts, 'NgramLengths', lengths)",
        inputs: &NGRAMS_COUNTS_REST,
        outputs: &OUTPUTS,
    },
];

pub const BAG_OF_NGRAMS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS_NGRAMS.INVALID_INPUT",
    identifier: Some("RunMat:bagOfNgrams:InvalidInput"),
    when: "Inputs do not match a supported bagOfNgrams form.",
    message: "bagOfNgrams: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [BAG_OF_NGRAMS_ERROR_INVALID_INPUT];

pub const BAG_OF_NGRAMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const COUNTS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "counts",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Explicit count matrices accept every integer class and require nonnegative values.",
}];
const LENGTHS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "lengths",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "NgramLengths accepts positive integer scalars or vectors from every integer class.",
    }];

pub const BAG_OF_NGRAMS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "bag = bagOfNgrams(uniqueNgrams, integer_counts, ...)",
        inputs: &COUNTS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Counts remain authoritative through nonnegative-integer validation and column filtering before one conversion into the model's documented double Counts property; the result is an opaque text-model object.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "bag = bagOfNgrams(..., \"NgramLengths\", integer_lengths)",
        inputs: &LENGTHS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Lengths are converted exactly into bounded host indices and returned as double object metadata; the result has no integer-class output.",
    },
];
