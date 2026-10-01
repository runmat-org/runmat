use crate::*;

const OUTPUT_IDX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based cluster index for each input observation.",
};

const OUTPUT_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Final cluster centroid matrix.",
};

const OUTPUT_SUMD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sumd",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Within-cluster sums of point-to-centroid distances.",
};

const OUTPUT_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Distance from every observation to every centroid.",
};

const OUTPUTS_IDX: [BuiltinParamDescriptor; 1] = [OUTPUT_IDX];
const OUTPUTS_IDX_C: [BuiltinParamDescriptor; 2] = [OUTPUT_IDX, OUTPUT_C];
const OUTPUTS_IDX_C_SUMD: [BuiltinParamDescriptor; 3] = [OUTPUT_IDX, OUTPUT_C, OUTPUT_SUMD];
const OUTPUTS_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_IDX, OUTPUT_C, OUTPUT_SUMD, OUTPUT_D];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of clusters.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Name-value options including Distance, Start, Replicates, MaxIter, EmptyAction, Display, OnlinePhase, and Options.",
};

const INPUTS_X_K: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_K];
const INPUTS_FULL: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_K, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "idx = kmeans(X, k)",
        inputs: &INPUTS_X_K,
        outputs: &OUTPUTS_IDX,
    },
    BuiltinSignatureDescriptor {
        label: "idx = kmeans(X, k, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX_C,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C, sumd] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX_C_SUMD,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C, sumd, D] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_FULL,
    },
];

pub const KMEANS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KMEANS.INVALID_ARGUMENT",
    identifier: Some("RunMat:kmeans:InvalidArgument"),
    when: "Inputs, cluster counts, starts, distance metrics, or name-value options are malformed.",
    message: "kmeans: invalid argument",
};

pub const KMEANS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KMEANS.INTERNAL",
    identifier: Some("RunMat:kmeans:Internal"),
    when: "RunMat cannot allocate or construct kmeans outputs.",
    message: "kmeans: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [KMEANS_ERROR_INVALID_ARGUMENT, KMEANS_ERROR_INTERNAL];

pub const KMEANS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const KMEANS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kmeans-integer-observation-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kmeans with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KmeansIntegerObservationDataExtension"),
};
pub const KMEANS_INTEGER_CLUSTER_COUNT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kmeans-integer-cluster-count",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kmeans with a typed-integer cluster count is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KmeansIntegerClusterCountExtension"),
    };
pub const KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kmeans-integer-iteration-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kmeans with typed-integer MaxIter or Replicates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KmeansIntegerIterationControlExtension"),
    };
pub const KMEANS_INTEGER_START_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kmeans-integer-start-centroids",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kmeans with typed-integer numeric start centroids is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KmeansIntegerStartCentroidsExtension"),
};
pub const KMEANS_INTEGER_PARALLEL_TOGGLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kmeans-integer-parallel-toggle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kmeans with typed-integer UseParallel or UseSubstreams is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KmeansIntegerParallelToggleExtension"),
    };

pub const KMEANS_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    KMEANS_INTEGER_DATA_EXTENSION,
    KMEANS_INTEGER_CLUSTER_COUNT_EXTENSION,
    KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
    KMEANS_INTEGER_START_EXTENSION,
    KMEANS_INTEGER_PARALLEL_TOGGLE_EXTENSION,
];

const INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The documented observation classes are single and double. RunMat admits typed integers only when every value is exactly representable at the binary64 clustering boundary.",
}];
const INTEGER_K_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "k",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The documented positive-integer control is stored as single or double; RunMat decodes a typed integer exactly as a bounded host cluster count.",
}];
const INTEGER_ITERATION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "MaxIter",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat decodes the positive iteration count directly from authoritative integer storage.",
    },
    BuiltinIntegerInputCapability {
        name: "Replicates",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat decodes the positive replicate count directly from authoritative integer storage.",
    },
];
const INTEGER_START_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Start",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Numeric Start is documented as single or double. Typed start centroids must be individually exact in binary64 before clustering.",
}];
const INTEGER_PARALLEL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Options.UseParallel",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "The documented field is logical; a typed numeric toggle is conservatively RunMat-only.",
    },
    BuiltinIntegerInputCapability {
        name: "Options.UseSubstreams",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "The documented field is logical; a typed numeric toggle is conservatively RunMat-only.",
    },
];

pub const KMEANS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[idx, C, sumd, D] = kmeans(integer_X, k, ...)",
        inputs: &INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Compatibility is decided before provider access; admitted integer data crosses one checked binary64 boundary. The current CPU fallback can return numeric outputs to the source provider when it supports them.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "kmeans(X, integer_k, ...)",
        inputs: &INTEGER_K_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "No floating conversion is used to decode the RunMat-only typed cluster count.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "kmeans(X, k, MaxIter/Replicates=integer_control)",
        inputs: &INTEGER_ITERATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Direct name-value and statset MaxIter forms share the same independently declared typed-integer control extension.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "kmeans(X, k, Start=integer_centroids)",
        inputs: &INTEGER_START_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The centroid array remains a numerical solver input, not a structural index, and therefore uses the checked binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "kmeans(X, k, Options.UseParallel/UseSubstreams=integer_toggle)",
        inputs: &INTEGER_PARALLEL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The current CPU implementation accepts only false for these unsupported parallel controls; typed zero is separately mode-gated.",
    },
];
