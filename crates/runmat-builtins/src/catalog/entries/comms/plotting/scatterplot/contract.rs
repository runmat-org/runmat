use crate::*;

const SCATTERPLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure/graphics handle returned by the underlying scatter plot.",
}];

const SCATTERPLOT_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex-baseband samples. Real inputs are plotted with zero imaginary part.",
}];

const SCATTERPLOT_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor; every n-th sample is plotted.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor; every n-th sample is plotted.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
    BuiltinParamDescriptor {
        name: "marker",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker LineSpec forwarded to scatter.",
    },
];

const SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_FIG: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex-baseband samples.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive decimation factor.",
    },
    BuiltinParamDescriptor {
        name: "offset",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-based sample offset before decimation.",
    },
    BuiltinParamDescriptor {
        name: "marker",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Marker LineSpec forwarded to scatter.",
    },
    BuiltinParamDescriptor {
        name: "scatfig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing figure handle used for the scatter plot.",
    },
];

const SCATTERPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x)",
        inputs: &SCATTERPLOT_INPUTS_X,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n)",
        inputs: &SCATTERPLOT_INPUTS_X_N,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset, marker)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterplot(x, n, offset, marker, scatfig)",
        inputs: &SCATTERPLOT_INPUTS_X_N_OFFSET_MARKER_FIG,
        outputs: &SCATTERPLOT_OUTPUT_HANDLE,
    },
];

pub const SCATTERPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatterplot:InvalidArgument"),
    when: "The input samples, decimation factor, offset, marker, or existing Figure handle is invalid.",
    message: "scatterplot: invalid argument",
};

pub const SCATTERPLOT_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERPLOT.PLOT_FAILED",
    identifier: Some("RunMat:scatterplot:PlotFailed"),
    when: "The underlying scatter plot cannot be rendered.",
    message: "scatterplot: plot operation failed",
};

const SCATTERPLOT_ERRORS: [BuiltinErrorDescriptor; 2] = [
    SCATTERPLOT_ERROR_INVALID_ARGUMENT,
    SCATTERPLOT_ERROR_PLOT_FAILED,
];

pub const SCATTERPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SCATTERPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SCATTERPLOT_ERRORS,
};

pub const SCATTERPLOT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterplot-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "scatterplot accepts typed-integer sample data as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterplotIntegerDataExtension"),
    };
pub const SCATTERPLOT_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterplot-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "scatterplot accepts typed-integer n and offset controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterplotIntegerControlExtension"),
    };
pub const SCATTERPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SCATTERPLOT_INTEGER_DATA_EXTENSION,
    SCATTERPLOT_INTEGER_CONTROL_EXTENSION,
];
const SCATTERPLOT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double samples; typed integer values require exact binary64 representation before constellation geometry is computed.",
    }];
const SCATTERPLOT_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target specifies a double positive-integer decimation factor; RunMat mode validates typed integer storage exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "offset",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target specifies a double nonnegative-integer offset; RunMat mode validates it exactly and requires offset < n.",
    },
];
pub const SCATTERPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterplot(integer_x, ...)",
        inputs: &SCATTERPLOT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The RunMat-only sample extension crosses into floating plotting coordinates only after exactness checks; the output is a double graphics handle.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterplot(x, integer_n [, integer_offset], ...)",
        inputs: &SCATTERPLOT_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Decimation controls remain exact usize-bounded integers and are never routed through binary64.",
    },
];
