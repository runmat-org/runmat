//! Canonical empirical-distribution descriptors, diagnostics, and capabilities.
use crate::*;
pub const ECDF_INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ecdf-integer-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ecdf with typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EcdfIntegerYExtension"),
};
pub const ECDF_LOGICAL_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ecdf-logical-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ecdf with logical sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EcdfLogicalYExtension"),
};
pub const ECDF_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ecdf-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ecdf with typed-integer Frequency data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EcdfIntegerFrequencyExtension"),
    };
pub const ECDF_LOGICAL_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ecdf-logical-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ecdf with logical Frequency data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EcdfLogicalFrequencyExtension"),
    };
pub const ECDF_INTEGER_CENSORING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ecdf-integer-censoring",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ecdf with typed-integer Censoring data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EcdfIntegerCensoringExtension"),
    };

pub const ECDF_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    ECDF_INTEGER_Y_EXTENSION,
    ECDF_LOGICAL_Y_EXTENSION,
    ECDF_INTEGER_FREQUENCY_EXTENSION,
    ECDF_LOGICAL_FREQUENCY_EXTENSION,
    ECDF_INTEGER_CENSORING_EXTENSION,
];

const ECDF_INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "y",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer samples are classified before provider access and must be exactly representable at the explicit binary64 statistics boundary.",
}];
const ECDF_INTEGER_FREQUENCY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Frequency",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer counts remain authoritative through validation and must be exactly representable at the floating ratio boundary.",
    }];
const ECDF_INTEGER_CENSORING_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Censoring",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer censoring indicators are validated exactly as -1, 0, or 1 before floating evaluation.",
    }];

pub const ECDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ecdf(y, ...) with typed-integer y",
        inputs: &ECDF_INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only explicit binary64 statistics boundary; all numeric outputs are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ecdf(y, 'Frequency', frequency, ...) with typed-integer frequency",
        inputs: &ECDF_INTEGER_FREQUENCY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Counts are checked as exact nonnegative integers before the floating ratio boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ecdf(y, 'Censoring', censoring, ...) with typed-integer censoring",
        inputs: &ECDF_INTEGER_CENSORING_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Indicators are exact structural labels; outputs remain floating statistics.",
    },
];

const OUTPUT_F: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empirical function values.",
};

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Evaluation points.",
};

const OUTPUT_FLO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flo",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower confidence bounds.",
};

const OUTPUT_FUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fup",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper confidence bounds.",
};

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the empirical CDF plot line.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Summary statistics structure.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data vector.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single- or double-precision sample-data vector.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Function, Censoring, Frequency, Alpha, and Bounds options.",
};

const INPUT_Y: [BuiltinParamDescriptor; 1] = [PARAM_Y];
const INPUT_Y_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_Y, PARAM_OPTIONS];
const INPUT_AX_Y: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_Y];
const INPUT_AX_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_Y, PARAM_OPTIONS];
const CDFPLOT_INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const CDFPLOT_INPUT_AX_X: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_X];
const OUTPUT_F_X: [BuiltinParamDescriptor; 2] = [OUTPUT_F, OUTPUT_X];
const OUTPUT_F_X_BOUNDS: [BuiltinParamDescriptor; 4] = [OUTPUT_F, OUTPUT_X, OUTPUT_FLO, OUTPUT_FUP];
const OUTPUT_H_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];
const OUTPUT_H_STATS: [BuiltinParamDescriptor; 2] = [OUTPUT_HANDLE, OUTPUT_STATS];

const ECDF_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "[f, x] = ecdf(y)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_F_X,
    },
    BuiltinSignatureDescriptor {
        label: "[f, x] = ecdf(y, Name, Value)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &OUTPUT_F_X,
    },
    BuiltinSignatureDescriptor {
        label: "[f, x, flo, fup] = ecdf(___)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &OUTPUT_F_X_BOUNDS,
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(y)",
        inputs: &INPUT_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(y, Name, Value)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(ax, y)",
        inputs: &INPUT_AX_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(ax, y, Name, Value)",
        inputs: &INPUT_AX_Y_OPTIONS,
        outputs: &[],
    },
];

const CDFPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "cdfplot(x)",
        inputs: &CDFPLOT_INPUT_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cdfplot(ax, x)",
        inputs: &CDFPLOT_INPUT_AX_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = cdfplot(___)",
        inputs: &CDFPLOT_INPUT_X,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[h, stats] = cdfplot(___)",
        inputs: &CDFPLOT_INPUT_X,
        outputs: &OUTPUT_H_STATS,
    },
];

pub const ECDF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.INVALID_ARGUMENT",
    identifier: Some("RunMat:ecdf:InvalidArgument"),
    when: "Inputs, name-value options, censoring, frequency, or alpha values are malformed.",
    message: "ecdf: invalid argument",
};

pub const ECDF_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.UNSUPPORTED",
    identifier: Some("RunMat:ecdf:Unsupported"),
    when: "A documented but not yet implemented interval or double-censored form is requested.",
    message: "ecdf: unsupported form",
};

pub const ECDF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.INTERNAL",
    identifier: Some("RunMat:ecdf:Internal"),
    when: "RunMat cannot construct ECDF outputs or plots.",
    message: "ecdf: internal error",
};

pub const CDFPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CDFPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:cdfplot:InvalidArgument"),
    when: "Inputs or axes arguments are malformed.",
    message: "cdfplot: invalid argument",
};

pub const CDFPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CDFPLOT.INTERNAL",
    identifier: Some("RunMat:cdfplot:Internal"),
    when: "RunMat cannot construct summary statistics or plots.",
    message: "cdfplot: internal error",
};

const ECDF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ECDF_ERROR_INVALID_ARGUMENT,
    ECDF_ERROR_UNSUPPORTED,
    ECDF_ERROR_INTERNAL,
];
const CDFPLOT_ERRORS: [BuiltinErrorDescriptor; 2] =
    [CDFPLOT_ERROR_INVALID_ARGUMENT, CDFPLOT_ERROR_INTERNAL];

pub const CDFPLOT_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cdfplot-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cdfplot with typed-integer sample data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CdfplotIntegerInputExtension"),
    };

pub const CDFPLOT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cdfplot-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cdfplot with logical sample data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CdfplotLogicalInputExtension"),
    };

pub const CDFPLOT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdfplot-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdfplot with gpuArray sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfplotGpuInputExtension"),
};

pub const CDFPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CDFPLOT_INTEGER_INPUT_EXTENSION,
    CDFPLOT_LOGICAL_INPUT_EXTENSION,
    CDFPLOT_GPU_INPUT_EXTENSION,
];

const CDFPLOT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dense typed-integer sample data is gated before provider access and must lie within the contiguous exact binary64 integer range at the explicit statistics and plotting boundary.",
    }];

pub const CDFPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "cdfplot(x) or cdfplot(ax, x) with dense integer sample data and any documented output form",
        inputs: &CDFPLOT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only visualization form; admitted integer observations cross one deliberate binary64 boundary, and outputs are a host graphics handle plus a structure of host double statistics.",
    }];

pub const ECDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ECDF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ECDF_ERRORS,
};

pub const CDFPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CDFPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CDFPLOT_ERRORS,
};
