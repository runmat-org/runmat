use crate::*;

const OUT_COMPOUND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "compoundScores",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Compound VADER-style sentiment score for each document.",
}];
const OUT_ALL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "compoundScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Compound VADER-style sentiment score for each document.",
    },
    BuiltinParamDescriptor {
        name: "positiveScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive sentiment proportions.",
    },
    BuiltinParamDescriptor {
        name: "negativeScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Negative sentiment proportions.",
    },
    BuiltinParamDescriptor {
        name: "neutralScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Neutral token proportions.",
    },
];
const IN_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];
const IN_DOCUMENTS_REST: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: SentimentLexicon, Boosters, Dampeners, Negations.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor { label: "compoundScores = vaderSentimentScores(documents)", inputs: &IN_DOCUMENTS, outputs: &OUT_COMPOUND },
    BuiltinSignatureDescriptor { label: "compoundScores = vaderSentimentScores(documents,Name,Value)", inputs: &IN_DOCUMENTS_REST, outputs: &OUT_COMPOUND },
    BuiltinSignatureDescriptor { label: "[compoundScores,positiveScores,negativeScores,neutralScores] = vaderSentimentScores(___)", inputs: &IN_DOCUMENTS_REST, outputs: &OUT_ALL },
];
pub const VADER_SENTIMENT_SCORES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.VADER_SENTIMENT_SCORES.INVALID_INPUT",
        identifier: Some("RunMat:vaderSentimentScores:InvalidInput"),
        when: "Inputs do not match supported vaderSentimentScores forms.",
        message: "vaderSentimentScores: invalid input",
    };
const ERRORS: [BuiltinErrorDescriptor; 1] = [VADER_SENTIMENT_SCORES_ERROR_INVALID_INPUT];
pub const VADER_SENTIMENT_SCORES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const VADER_INTEGER_SCORE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SentimentLexicon.SentimentScore", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented numeric scalar scores may use any integer class. Their [-4,4] range is checked on authoritative storage before exact conversion to floating VADER computation.",
}];
pub const VADER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "scores = vaderSentimentScores(documents, SentimentLexicon=table(integer_scores))",
    inputs: &VADER_INTEGER_SCORE_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Every valid integer lexicon score is exactly representable as binary64 because its public domain is [-4,4]. Supported resident table payloads may gather through their owner; outputs are host double score vectors.",
}];
