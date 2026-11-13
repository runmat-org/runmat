use crate::*;

pub const COSINE_SIMILARITY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.COSINE_SIMILARITY.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:cosineSimilarity:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "cosineSimilarity: too many output arguments",
    };

pub const COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-integer-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with typed-integer matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityIntegerMatrixExtension"),
    };

pub const COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-single-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with single matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilaritySingleMatrixExtension"),
    };

pub const COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-logical-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with logical matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityLogicalMatrixExtension"),
    };

pub const COSINE_SIMILARITY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with resident matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityResidentInputExtension"),
    };

pub const COSINE_SIMILARITY_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION,
    COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION,
    COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION,
    COSINE_SIMILARITY_RESIDENT_INPUT_EXTENSION,
];

const COSINE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "M1", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Real or typed-complex integer matrix values must be exactly representable at the binary64 norm and inner-product boundary." },
    BuiltinIntegerInputCapability { name: "M2", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The optional second matrix follows the same exact binary64 boundary and column-count rule." },
];

pub const COSINE_SIMILARITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "similarities = cosineSimilarity(integer_M1, integer_M2?)", inputs: &COSINE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat mode validates authoritative integer components before double norm/dot evaluation; real output is host sparse double, while complex output remains a dense complex-double representation gap." }];

const OUT_SIMILARITIES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "similarities",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise cosine-similarity matrix.",
}];

const IN_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrBagOrMatrix",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object, bag model, word vector, or numeric matrix.",
}];

const IN_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrBagOrMatrix",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object, bag model, word vector, or numeric matrix.",
    },
    BuiltinParamDescriptor {
        name: "queriesOrMatrix",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query tokenizedDocument object, word vector, or numeric matrix.",
    },
];

pub const COSINE_SIMILARITY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSINE_SIMILARITY.INVALID_INPUT",
    identifier: Some("RunMat:cosineSimilarity:InvalidInput"),
    when: "Inputs do not match a supported cosineSimilarity form.",
    message: "cosineSimilarity: invalid input",
};

pub const COSINE_SIMILARITY_ERROR_DIMENSION_MISMATCH: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.COSINE_SIMILARITY.DIMENSION_MISMATCH",
        identifier: Some("RunMat:cosineSimilarity:DimensionMismatch"),
        when: "Matrix inputs have different numbers of columns.",
        message: "cosineSimilarity: matrix dimensions are not compatible",
    };

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    COSINE_SIMILARITY_ERROR_INVALID_INPUT,
    COSINE_SIMILARITY_ERROR_DIMENSION_MISMATCH,
    COSINE_SIMILARITY_ERROR_TOO_MANY_OUTPUTS,
];

pub const COSINE_SIMILARITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(documents)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(documents, queries)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(bag)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(bag, queries)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(M)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(M1, M2)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
