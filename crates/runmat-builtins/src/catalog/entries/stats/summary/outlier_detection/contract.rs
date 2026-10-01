//! Shared public signatures, errors, and independently gated forms.
use crate::*;
const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric array.",
};

const PARAM_FILL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fillmethod",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Replacement method or constant value.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Detection method, dimension, and name-value options.",
};

const OUTPUT_MASK: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask of detected outliers.",
};

const OUTPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with outliers filled.",
};

const OUTPUT_LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower threshold.",
};

const OUTPUT_UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "U",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper threshold.",
};

const OUTPUT_CENTER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Center value used by the detection method.",
};

const INPUT_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUT_A_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_OPTIONS];
const INPUT_A_FILL: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_FILL];
const INPUT_A_FILL_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_FILL, PARAM_OPTIONS];
const OUT_MASK: [BuiltinParamDescriptor; 1] = [OUTPUT_MASK];
const OUT_MASK_THRESHOLDS: [BuiltinParamDescriptor; 4] =
    [OUTPUT_MASK, OUTPUT_LOWER, OUTPUT_UPPER, OUTPUT_CENTER];
const OUT_VALUE: [BuiltinParamDescriptor; 1] = [OUTPUT_VALUE];
const OUT_VALUE_MASK: [BuiltinParamDescriptor; 2] = [OUTPUT_VALUE, OUTPUT_MASK];
const OUT_VALUE_THRESHOLDS: [BuiltinParamDescriptor; 5] = [
    OUTPUT_VALUE,
    OUTPUT_MASK,
    OUTPUT_LOWER,
    OUTPUT_UPPER,
    OUTPUT_CENTER,
];

const ISOUTLIER_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A)",
        inputs: &INPUT_A,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A, method)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A, ___, dim)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[TF, L, U, C] = isoutlier(___)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK_THRESHOLDS,
    },
];

const FILLOUTLIERS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, fillmethod)",
        inputs: &INPUT_A_FILL,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, fillmethod, findmethod)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, ___, dim)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, TF] = filloutliers(___)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[B, TF, L, U, C] = filloutliers(___)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE_THRESHOLDS,
    },
];

pub const OUTLIER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.INVALID_ARGUMENT",
    identifier: Some("RunMat:outliers:InvalidArgument"),
    when: "Inputs, methods, dimensions, or name-value options are malformed.",
    message: "outlier builtin: invalid argument",
};

pub const OUTLIER_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.UNSUPPORTED",
    identifier: Some("RunMat:outliers:Unsupported"),
    when: "A table/timetable or advanced smoothing/interpolation form is requested.",
    message: "outlier builtin: unsupported form",
};

pub const OUTLIER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.INTERNAL",
    identifier: Some("RunMat:outliers:Internal"),
    when: "Internal tensor or logical array materialization fails.",
    message: "outlier builtin: internal error",
};

pub const OUTLIER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:outliers:TooManyOutputs"),
    when: "More outputs are requested than the builtin can return.",
    message: "outlier builtin: too many outputs",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    OUTLIER_ERROR_INVALID_ARGUMENT,
    OUTLIER_ERROR_UNSUPPORTED,
    OUTLIER_ERROR_INTERNAL,
    OUTLIER_ERROR_TOO_MANY_OUTPUTS,
];

pub const ISOUTLIER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISOUTLIER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ISOUTLIER_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with typed-integer input data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerDataExtension"),
    };
pub const ISOUTLIER_INTEGER_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with a typed-integer dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerDimensionExtension"),
    };
pub const ISOUTLIER_INTEGER_WINDOW_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-window",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with a typed-integer moving window is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerWindowExtension"),
    };
pub const ISOUTLIER_INTEGER_THRESHOLD_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-threshold",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with typed-integer percentile or ThresholdFactor values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerThresholdExtension"),
    };
pub const ISOUTLIER_GPU_MOVMEDIAN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-gpu-movmedian",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "an explicit GPU isoutlier call using movmedian is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierGpuMovmedianExtension"),
    };
pub const ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-gpu-sample-points",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "an explicit GPU isoutlier call using SamplePoints is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierGpuSamplePointsExtension"),
    };
pub const ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-gpu-data-variables",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "an explicit GPU isoutlier call using DataVariables is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierGpuDataVariablesExtension"),
    };

pub const ISOUTLIER_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    ISOUTLIER_INTEGER_DATA_EXTENSION,
    ISOUTLIER_INTEGER_DIMENSION_EXTENSION,
    ISOUTLIER_INTEGER_WINDOW_EXTENSION,
    ISOUTLIER_INTEGER_THRESHOLD_EXTENSION,
    ISOUTLIER_GPU_MOVMEDIAN_EXTENSION,
    ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
    ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
];

const ISOUTLIER_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric data classes are single and double. Each RunMat-only typed value must be exactly representable at the binary64 statistics boundary.",
    }];
const ISOUTLIER_INTEGER_STRUCTURAL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation specifies a positive integer value domain but does not enumerate native integer storage classes, so typed storage remains conservatively RunMat-only.",
    },
    BuiltinIntegerInputCapability {
        name: "window",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Count-form moving windows are decoded exactly; duration windows are a separate, currently unsupported surface.",
    },
];
const ISOUTLIER_INTEGER_THRESHOLD_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "threshold",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed percentile bounds are not publicly class-enumerated and must be exactly representable at the floating percentile boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "ThresholdFactor",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed threshold factors are not publicly class-enumerated and must be exactly representable at the floating statistics boundary.",
    },
];

pub const ISOUTLIER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[TF, L, U, C] = isoutlier(integer_A, ...)",
        inputs: &ISOUTLIER_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "TF is logical; threshold outputs are floating. Compatibility and exactness are checked before provider access, and the current CPU fallback may restore supported outputs to the source provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "isoutlier(A, ..., integer_dim/window)",
        inputs: &ISOUTLIER_INTEGER_STRUCTURAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "These ambiguous public storage-class forms stay independently RunMat-gated rather than being overclaimed as documented native-integer support.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "isoutlier(A, percentiles, integer_threshold / ThresholdFactor=integer_factor)",
        inputs: &ISOUTLIER_INTEGER_THRESHOLD_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Ambiguous typed storage is admitted only in RunMat mode and only when every value is exact in binary64.",
    },
];

pub const FILLOUTLIERS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILLOUTLIERS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FILLOUTLIERS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filloutliers with typed-integer input data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersIntegerDataExtension"),
    };
pub const FILLOUTLIERS_INTEGER_FILL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-integer-fill-scalar",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filloutliers with a typed-integer constant fill scalar is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersIntegerFillExtension"),
    };
pub const FILLOUTLIERS_NUMERIC_MASK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-numeric-outlier-locations",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "filloutliers with numeric rather than logical OutlierLocations is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersNumericMaskExtension"),
    };
pub const FILLOUTLIERS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "filloutliers with a direct provider-resident input or argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersResidentInputExtension"),
    };
pub const FILLOUTLIERS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FILLOUTLIERS_INTEGER_DATA_EXTENSION,
    FILLOUTLIERS_INTEGER_FILL_EXTENSION,
    FILLOUTLIERS_NUMERIC_MASK_EXTENSION,
    FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
];

const FILLOUTLIERS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight classes are admitted only when the exact input span is at most 2^53. Wider spans are rejected before statistics rather than collapsing distinct observations at the double boundary.",
    }];
const FILLOUTLIERS_INTEGER_FILL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fillmethod",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A typed-integer scalar constant is decoded from authoritative storage and converted once to the output computation domain.",
    }];
const FILLOUTLIERS_INTEGER_MASK_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "OutlierLocations",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented masks are logical; numeric masks are accepted only in RunMat mode and must exactly match A's shape.",
    }];
pub const FILLOUTLIERS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B, TF, L, U, C] = filloutliers(integer_A, ...)",
        inputs: &FILLOUTLIERS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "For admitted inputs, exact anchored integer differences are representable in binary64. Inputs spanning more than 2^53 are rejected; B, L, U, and C deliberately cross to double, while TF is logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "filloutliers(A, integer_fill_scalar, ...)",
        inputs: &FILLOUTLIERS_INTEGER_FILL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed-integer constant-fill role is independently mode-gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "filloutliers(A, ..., OutlierLocations=integer_mask)",
        inputs: &FILLOUTLIERS_INTEGER_MASK_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Zero is false and nonzero is true after the independently gated numeric-mask extension is admitted.",
    },
];
