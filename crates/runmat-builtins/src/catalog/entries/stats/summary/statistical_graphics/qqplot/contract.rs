//! Public quantile-plot forms, diagnostic ownership and numeric extensions.
use crate::*;

const OUTPUT_HANDLES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line handles for the quantile points, quartile line, and fitted reference line.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data vector or matrix.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional comparison sample data vector.",
};

const PARAM_PVEC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pvec",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional quantile percentages in the closed interval [0, 100].",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_X_Y_P: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_PVEC];
const INPUT_AX_X: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_X];
const INPUT_AX_X_Y_P: [BuiltinParamDescriptor; 4] = [PARAM_AX, PARAM_X, PARAM_Y, PARAM_PVEC];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLES];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "qqplot(x)",
        inputs: &INPUT_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(x, y)",
        inputs: &INPUT_X_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(x, y, pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(x, [], pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "qqplot(ax, ___)",
        inputs: &INPUT_AX_X,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(___)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(x, y, pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(x, [], pvec)",
        inputs: &INPUT_X_Y_P,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = qqplot(ax, ___)",
        inputs: &INPUT_AX_X_Y_P,
        outputs: &OUTPUT_H,
    },
];

pub const QQPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QQPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:qqplot:InvalidArgument"),
    when: "Inputs, probabilities, or axes handles are malformed.",
    message: "qqplot: invalid argument",
};

pub const QQPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QQPLOT.INTERNAL",
    identifier: Some("RunMat:qqplot:Internal"),
    when: "RunMat cannot construct quantile pairs or plotting primitives.",
    message: "qqplot: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [QQPLOT_ERROR_INVALID_ARGUMENT, QQPLOT_ERROR_INTERNAL];

pub const QQPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const QQPLOT_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "qqplot-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "qqplot accepts typed-integer sample data as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:QqplotIntegerSampleExtension"),
    };
pub const QQPLOT_INTEGER_PVEC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-integer-pvec",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts a typed-integer quantile vector as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotIntegerPvecExtension"),
};
pub const QQPLOT_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts explicitly GPU-resident inputs as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotExplicitGpuInputExtension"),
};
pub const QQPLOT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qqplot-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qqplot accepts logical sample or percentage input as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QqplotLogicalInputExtension"),
};
pub const QQPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    QQPLOT_INTEGER_SAMPLE_EXTENSION,
    QQPLOT_INTEGER_PVEC_EXTENSION,
    QQPLOT_EXPLICIT_GPU_EXTENSION,
    QQPLOT_LOGICAL_INPUT_EXTENSION,
];

/// Semantic role after the optional axes handle has been removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QqplotInputRole {
    SampleX,
    SampleY,
    Percentages,
}

impl QqplotInputRole {
    pub const fn at(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::SampleX),
            1 => Some(Self::SampleY),
            2 => Some(Self::Percentages),
            _ => None,
        }
    }

    pub const fn integer_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::SampleX | Self::SampleY => &QQPLOT_INTEGER_SAMPLE_EXTENSION,
            Self::Percentages => &QQPLOT_INTEGER_PVEC_EXTENSION,
        }
    }

    pub const fn integer_label(self) -> &'static str {
        match self {
            Self::SampleX | Self::SampleY => "sample",
            Self::Percentages => "pvec",
        }
    }
}
const QQPLOT_INTEGER_SAMPLES: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample data. RunMat admits typed integers only after exact binary64 representability is proved.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The optional second sample has the same independently gated checked floating boundary as x.",
    },
];
const QQPLOT_INTEGER_PVEC: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "pvec",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single and double quantile percentages; typed integer percentages cross a checked binary64 normalization boundary.",
}];
pub const QQPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = qqplot(integer_x [, integer_y])",
        inputs: &QQPLOT_INTEGER_SAMPLES,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Quantiles, interpolation, and reference-line geometry are explicit floating boundaries. Automatic residency gathers transparently; explicit undocumented GPU intent is independently gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = qqplot(x,y,integer_pvec)",
        inputs: &QQPLOT_INTEGER_PVEC,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed quantile percentages are a separately gated RunMat extension and are normalized only after exactness and range checks.",
    },
];
