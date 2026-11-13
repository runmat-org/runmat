use crate::*;

const PLOT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X/Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Coordinate data accept all eight integer classes and retain native class, shape, and exact values as graphics-object source properties.",
    }];
pub const PLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = plot([X,] integer_Y, ...)",
        inputs: &PLOT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native numeric storage remains authoritative for XData/YData and figure persistence. Rendering is an explicit floating geometry boundary; integer gpuArray inputs gather exactly because the direct WGPU line path is floating-only.",
    }];

const PLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the first line object in the rendered plot series.",
}];

const PLOT_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y data; X defaults to 1:numel(Y).",
}];

const PLOT_INPUTS_Y_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data; X defaults to 1:numel(Y).",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style/color shorthand.",
    },
];

const PLOT_INPUTS_Y_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data; X defaults to 1:numel(Y).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties.",
    },
];

const PLOT_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
];

const PLOT_INPUTS_X_Y_STYLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style/color shorthand.",
    },
];

const PLOT_INPUTS_X_Y_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties.",
    },
];

const PLOT_INPUTS_AX_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data; X defaults to 1:numel(Y).",
    },
];

const PLOT_INPUTS_AX_X_Y: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
];

const PLOT_INPUTS_AX_X_Y_STYLE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
    BuiltinParamDescriptor {
        name: "lineSpec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style/color shorthand.",
    },
];

const PLOT_INPUTS_AX_X_Y_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style properties.",
    },
];

const PLOT_INPUTS_MULTI_SERIES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "series",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Interleaved X/Y series pairs with optional per-series style tokens.",
}];

const PLOT_INPUTS_AX_MULTI_SERIES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "series",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Interleaved X/Y series pairs with optional per-series style tokens.",
    },
];

const PLOT_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h = plot(Y)",
        inputs: &PLOT_INPUTS_Y,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(Y, LineSpec)",
        inputs: &PLOT_INPUTS_Y_STYLE,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(Y, Name, Value, ...)",
        inputs: &PLOT_INPUTS_Y_PROPS,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(X, Y)",
        inputs: &PLOT_INPUTS_X_Y,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(X, Y, LineSpec)",
        inputs: &PLOT_INPUTS_X_Y_STYLE,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(X, Y, Name, Value, ...)",
        inputs: &PLOT_INPUTS_X_Y_PROPS,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(ax, Y)",
        inputs: &PLOT_INPUTS_AX_Y,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(ax, X, Y)",
        inputs: &PLOT_INPUTS_AX_X_Y,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(ax, X, Y, LineSpec)",
        inputs: &PLOT_INPUTS_AX_X_Y_STYLE,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(ax, X, Y, Name, Value, ...)",
        inputs: &PLOT_INPUTS_AX_X_Y_PROPS,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(X1, Y1, ..., Xn, Yn)",
        inputs: &PLOT_INPUTS_MULTI_SERIES,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot(ax, X1, Y1, ..., Xn, Yn)",
        inputs: &PLOT_INPUTS_AX_MULTI_SERIES,
        outputs: &PLOT_OUTPUT_HANDLE,
    },
];

pub const PLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:plot:InvalidArgument"),
    when: "Input data, series grammar, axes targeting, or style arguments are invalid.",
    message: "plot: invalid argument",
};

pub const PLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOT.INTERNAL",
    identifier: Some("RunMat:plot:Internal"),
    when: "Internal line construction or rendering fails unexpectedly.",
    message: "plot: internal operation failed",
};

pub const PLOT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plot:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "plot: too many output arguments",
};
const PLOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PLOT_ERROR_INVALID_ARGUMENT,
    PLOT_ERROR_INTERNAL,
    PLOT_ERROR_TOO_MANY_OUTPUTS,
];

pub const PLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PLOT_ERRORS,
};
