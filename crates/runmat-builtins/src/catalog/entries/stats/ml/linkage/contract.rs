use crate::*;

pub const LINKAGE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "linkage-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "linkage with typed-integer observation or distance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LinkageIntegerDataExtension"),
};

pub const LINKAGE_INTEGER_DISTANCE_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linkage-integer-distance-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linkage with typed-integer pdist distance parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinkageIntegerDistanceParameterExtension"),
    };

pub const LINKAGE_EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "linkage-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "linkage host fallback for explicit gpuArray inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LinkageExplicitGpuInputExtension"),
    };

pub const LINKAGE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    LINKAGE_INTEGER_DATA_EXTENSION,
    LINKAGE_INTEGER_DISTANCE_PARAMETER_EXTENSION,
    LINKAGE_EXPLICIT_GPU_INPUT_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X or y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer observations or condensed distances are gated before gather and checked for exact binary64 representation.",
    }];

const INTEGER_DISTANCE_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "pdist DistParameter",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer Minkowski exponents, standardized scales, and covariance entries are checked before entering floating distance arithmetic.",
    }];

pub const LINKAGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = linkage(integer_X_or_y, ___)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer data crosses a checked binary64 distance boundary. Z is a double matrix: columns 1 and 2 are integer-valued cluster labels and column 3 is merge distance.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Z = linkage(X, method, {metric, integer_DistParameter})",
        inputs: &INTEGER_DISTANCE_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Distance parameters are numeric algorithm inputs, not structural integer controls; lossy integer-to-double conversion rejects.",
    },
];

const OUTPUT_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hierarchical cluster tree with one merge per row.",
}];

const PARAM_X_OR_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X_or_Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix X or condensed distance vector Y.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Linkage method, distance metric, pdist metric parameters, and SaveMemory option.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 1] = [PARAM_X_OR_Y];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X_OR_Y, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, metric)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, metric, 'savememory', value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(X, method, pdist_inputs)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = linkage(Y, method)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUT_Z,
    },
];

pub const LINKAGE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:linkage:InvalidArgument"),
    when: "Inputs, linkage methods, distance metrics, or name-value options are malformed.",
    message: "linkage: invalid argument",
};

pub const LINKAGE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAGE.INTERNAL",
    identifier: Some("RunMat:linkage:Internal"),
    when: "RunMat cannot allocate or construct the linkage output.",
    message: "linkage: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [LINKAGE_ERROR_INVALID_ARGUMENT, LINKAGE_ERROR_INTERNAL];

pub const LINKAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
