use crate::*;

const OUTPUT_IDX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Idx",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based nearest-row indices in X, or cell array of index vectors when IncludeTies is true.",
};
const OUTPUT_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Distances to nearest rows in X, or cell array of distance vectors when IncludeTies is true.",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};
const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Query matrix with query observations in rows.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options K, Distance, P, Cov, Scale, IncludeTies, NSMethod, BucketSize, CacheSize, and SortIndices.",
};
const OUTPUTS: [BuiltinParamDescriptor; 2] = [OUTPUT_IDX, OUTPUT_D];
const INPUTS: [BuiltinParamDescriptor; 2] = [X, Y];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [X, Y, OPTIONS];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Idx = knnsearch(X, Y)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[Idx, D] = knnsearch(X, Y, Name, Value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS,
    },
];
pub const KNNSEARCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::distance_errors::DISTANCE_ERRORS,
};

pub const KNN_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "knnsearch-integer-observation-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "knnsearch with typed-integer X or Y data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KnnsearchIntegerObservationDataExtension"),
};
pub const KNN_INTEGER_STRUCTURAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "knnsearch-integer-structural-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "knnsearch with typed-integer K or BucketSize is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KnnsearchIntegerStructuralControlExtension"),
    };
pub const KNN_INTEGER_FLOATING_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "knnsearch-integer-floating-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "knnsearch with typed-integer P, Cov, Scale, or CacheSize is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KnnsearchIntegerFloatingControlExtension"),
    };
pub const KNN_INTEGER_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "knnsearch-integer-logical-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "knnsearch with typed-integer IncludeTies or SortIndices is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KnnsearchIntegerLogicalControlExtension"),
    };
pub const KNN_GPU_NSMETHOD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "knnsearch-gpu-nonexhaustive-search",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "an explicit GPU knnsearch call without NSMethod='exhaustive' is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KnnsearchGpuNonexhaustiveSearchExtension"),
};
pub const KNN_GPU_INCLUDE_TIES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "knnsearch-gpu-include-ties",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "an explicit GPU knnsearch call with nondefault IncludeTies is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KnnsearchGpuIncludeTiesExtension"),
};
pub const KNN_GPU_SORT_INDICES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "knnsearch-gpu-sort-indices",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "an explicit GPU knnsearch call with nondefault SortIndices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KnnsearchGpuSortIndicesExtension"),
};
pub const KNN_GPU_FAST_DISTANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "knnsearch-gpu-fast-distance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "an explicit GPU knnsearch call with a fast distance variant is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KnnsearchGpuFastDistanceExtension"),
    };
pub const KNNSEARCH_EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    KNN_INTEGER_DATA_EXTENSION,
    KNN_INTEGER_STRUCTURAL_EXTENSION,
    KNN_INTEGER_FLOATING_CONTROL_EXTENSION,
    KNN_INTEGER_LOGICAL_CONTROL_EXTENSION,
    KNN_GPU_NSMETHOD_EXTENSION,
    KNN_GPU_INCLUDE_TIES_EXTENSION,
    KNN_GPU_SORT_INDICES_EXTENSION,
    KNN_GPU_FAST_DISTANCE_EXTENSION,
];

const DATA: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every admitted integer value must be exact at the binary64 distance boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every admitted integer value must be exact at the binary64 distance boundary.",
    },
];
const STRUCTURAL: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "K",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "K is decoded exactly as a positive output-shape count.",
    },
    BuiltinIntegerInputCapability {
        name: "BucketSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "BucketSize is decoded exactly as a positive search-structure count.",
    },
];
const FLOATING: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "P",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The Minkowski exponent crosses a checked binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Cov",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Covariance crosses a checked binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Scale",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Scale crosses a checked binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "CacheSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "CacheSize crosses a checked binary64 boundary.",
    },
];
const LOGICAL: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "IncludeTies",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Only exact zero and one are accepted.",
    },
    BuiltinIntegerInputCapability {
        name: "SortIndices",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Only exact zero and one are accepted.",
    },
];
pub const KNNSEARCH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[Idx,D] = knnsearch(integer_X, integer_Y, ...)",
        inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Idx and D are double. Host fallback can restore numeric outputs to the source provider when supported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "knnsearch(X,Y,K/BucketSize=integer_control)",
        inputs: &STRUCTURAL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Structural controls decode directly from native storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "knnsearch(X,Y,P/Cov/Scale/CacheSize=integer_value)",
        inputs: &FLOATING,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed numerical controls must be exactly representable in binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "knnsearch(X,Y,IncludeTies/SortIndices=integer_toggle)",
        inputs: &LOGICAL,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact zero and one are accepted after the RunMat extension is admitted.",
    },
];
