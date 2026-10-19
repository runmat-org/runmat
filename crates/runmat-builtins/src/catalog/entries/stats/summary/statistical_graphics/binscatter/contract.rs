//! Binned-scatter signatures, diagnostics and exact integer capabilities.
use crate::*;

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the Binscatter chart.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X coordinates or table variable selector.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y coordinates or table variable selector.",
};

const PARAM_XVAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "xvar",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table variable name or positive one-based variable index for X.",
};

const PARAM_YVAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yvar",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table variable name or positive one-based variable index for Y.",
};

const PARAM_TABLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or timetable.",
};

const PARAM_NBINS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nbins",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Scalar or two-element bin count.",
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
    description: "Binscatter name-value options.",
};

const INPUT_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_XY_N: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_NBINS];
const INPUT_XY_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];
const INPUT_TBL_XY: [BuiltinParamDescriptor; 3] = [PARAM_TABLE, PARAM_XVAR, PARAM_YVAR];
const INPUT_TBL_XY_N: [BuiltinParamDescriptor; 4] =
    [PARAM_TABLE, PARAM_XVAR, PARAM_YVAR, PARAM_NBINS];
const INPUT_TBL_XY_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_TABLE, PARAM_XVAR, PARAM_YVAR, PARAM_OPTIONS];
const INPUT_AX_XY: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_X, PARAM_Y];
const INPUT_AX_XY_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_X, PARAM_Y, PARAM_OPTIONS];
const INPUT_AX_TBL_XY: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_TABLE, PARAM_XVAR, PARAM_YVAR];
const OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];

const SIGNATURES: [BuiltinSignatureDescriptor; 14] = [
    BuiltinSignatureDescriptor {
        label: "binscatter(x, y)",
        inputs: &INPUT_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(x, y, nbins)",
        inputs: &INPUT_XY_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(___, Name, Value)",
        inputs: &INPUT_XY_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(tbl, xvar, yvar)",
        inputs: &INPUT_TBL_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(tbl, xvar, yvar, nbins)",
        inputs: &INPUT_TBL_XY_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(tbl, xvar, yvar, Name, Value)",
        inputs: &INPUT_TBL_XY_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(ax, ___)",
        inputs: &INPUT_AX_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(ax, x, y, Name, Value)",
        inputs: &INPUT_AX_XY_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "binscatter(ax, tbl, xvar, yvar)",
        inputs: &INPUT_AX_TBL_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(___)",
        inputs: &INPUT_XY,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(x, y, nbins)",
        inputs: &INPUT_XY_N,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(x, y, Name, Value)",
        inputs: &INPUT_XY_OPTIONS,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(tbl, xvar, yvar, nbins)",
        inputs: &INPUT_TBL_XY_N,
        outputs: &OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = binscatter(ax, x, y, Name, Value)",
        inputs: &INPUT_AX_XY_OPTIONS,
        outputs: &OUTPUT_H,
    },
];

pub const BINSCATTER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:binscatter:InvalidArgument"),
    when: "Inputs, bin controls, or chart options are malformed.",
    message: "binscatter: invalid argument",
};

pub const BINSCATTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BINSCATTER.INTERNAL",
    identifier: Some("RunMat:binscatter:Internal"),
    when: "RunMat cannot compute bins or render the chart.",
    message: "binscatter: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BINSCATTER_ERROR_INVALID_ARGUMENT, BINSCATTER_ERROR_INTERNAL];

pub const BINSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BINSCATTER_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binscatter-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binscatter with resident GPU-array inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinscatterGpuInputExtension"),
};

pub const BINSCATTER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BINSCATTER_GPU_INPUT_EXTENSION];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented real vector X data accepts every built-in integer class and remains authoritative through vector validation and chart state.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented real vector Y data accepts every built-in integer class and remains authoritative through vector validation and chart state.",
    },
];

const INTEGER_BIN_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "N/NumBins",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented scalar or two-element bin counts accept every built-in integer class and are parsed exactly into bounded host dimensions.",
    }];

const INTEGER_LIMIT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "XLimits",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented two-element X limits accept every built-in integer class and are validated exactly before the floating bin-edge and graphics boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "YLimits",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented two-element Y limits accept every built-in integer class and are validated exactly before the floating bin-edge and graphics boundary.",
    },
];

const INTEGER_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "xvar",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A documented positive numeric scalar can select the X table variable; typed integers are parsed exactly as one-based structural indices.",
    },
    BuiltinIntegerInputCapability {
        name: "yvar",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A documented positive numeric scalar can select the Y table variable; typed integers are parsed exactly as one-based structural indices.",
    },
];

const INTEGER_TOGGLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ShowEmptyBins",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented numeric zero-or-one toggle accepts every scalar integer class and is parsed exactly.",
    }];

const INTEGER_ALPHA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FaceAlpha",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented numeric scalar transparency control admits integer zero or one and crosses once into host graphics state.",
    }];

pub const BINSCATTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(integer_x, integer_y, ...)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer XData/YData class, shape, and values are retained in the opaque chart state; exact rational bin membership and relative normalization precede floating bin-edge and client-graphics construction. Resident input is a separately gated RunMat extension.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(x, y, integer_N)",
        inputs: &INTEGER_BIN_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar N expands to both dimensions; a two-element vector selects each dimension; every count must be in 1..=250.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(..., XLimits/YLimits, integer_limits)",
        inputs: &INTEGER_LIMIT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Same-axis integer data and limits use exact i128 rational bin membership across the full built-in integer range, with relative normalization only for floating bin-edge rendering.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(tbl, integer_xvar, integer_yvar, ...)",
        inputs: &INTEGER_SELECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Positive scalar integer selectors resolve table variables without floating conversion; the selected data retains its own documented class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(..., \"ShowEmptyBins\", integer_toggle)",
        inputs: &INTEGER_TOGGLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Only exact zero and one are accepted; chart state exposes the resulting logical property.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = binscatter(..., \"FaceAlpha\", integer_alpha)",
        inputs: &INTEGER_ALPHA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer transparency is necessarily zero or one under the documented inclusive unit interval and enters one host f64 graphics boundary.",
    },
];
