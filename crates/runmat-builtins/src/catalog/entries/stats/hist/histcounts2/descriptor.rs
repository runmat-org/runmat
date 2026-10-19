use crate::*;
const HISTCOUNTS2_OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D histogram bin counts.",
}];

const HISTCOUNTS2_OUTPUT_NX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
];

const HISTCOUNTS2_OUTPUT_NXY: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "Yedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin edges.",
    },
];

const HISTCOUNTS2_OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "Yedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "binX",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based X-bin indices, or zero for unbinned observations.",
    },
    BuiltinParamDescriptor {
        name: "binY",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based Y-bin indices, or zero for unbinned observations.",
    },
];

const HISTCOUNTS2_INPUTS_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
];

const HISTCOUNTS2_INPUTS_XY_BINS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "NumBins scalar/vector or positional edge specification.",
    },
];

const HISTCOUNTS2_INPUTS_XY_BINS_BINSY: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis NumBins scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis NumBins scalar or edge vector.",
    },
];

const HISTCOUNTS2_INPUTS_XY_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value pairs controlling binning and normalization.",
    },
];

const HISTCOUNTS2_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, bins)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, binsX, binsY)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS_BINSY,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_NX,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges] = histcounts2(X, Y, bins)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS,
        outputs: &HISTCOUNTS2_OUTPUT_NX,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y, binsX, binsY)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS_BINSY,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges, binX, binY] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges, binX, binY] = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_ALL,
    },
];

pub const HISTCOUNTS2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.INVALID_ARGUMENT",
    identifier: Some("RunMat:histcounts2:InvalidArgument"),
    when: "Arguments are malformed, inconsistent, or unsupported.",
    message: "histcounts2: invalid argument",
};

pub const HISTCOUNTS2_ERROR_NUMBINS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.NUMBINS_INVALID",
    identifier: Some("RunMat:histcounts2:NumBinsInvalid"),
    when: "NumBins is zero, negative, non-finite, or non-integer.",
    message: "histcounts2: NumBins must be a positive finite scalar",
};

pub const HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.BINMETHOD_CONFLICT",
    identifier: Some("RunMat:histcounts2:BinMethodConflict"),
    when: "BinMethod is combined with incompatible bin controls.",
    message: "histcounts2: BinMethod cannot be combined with BinEdges, NumBins, or BinWidth",
};

pub const HISTCOUNTS2_ERROR_EDGE_PRECISION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.EDGE_PRECISION",
    identifier: Some("RunMat:histcounts2:EdgePrecision"),
    when: "Integer-aligned unit bin boundaries cannot be represented distinctly by the double edge output.",
    message: "histcounts2: integer-aligned bin boundaries cannot be represented distinctly as double",
};

pub const HISTCOUNTS2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.INTERNAL",
    identifier: Some("RunMat:histcounts2:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "histcounts2: internal operation failed",
};

const HISTCOUNTS2_ERRORS: [BuiltinErrorDescriptor; 5] = [
    HISTCOUNTS2_ERROR_INVALID_ARGUMENT,
    HISTCOUNTS2_ERROR_NUMBINS_INVALID,
    HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT,
    HISTCOUNTS2_ERROR_EDGE_PRECISION,
    HISTCOUNTS2_ERROR_INTERNAL,
];

pub const HISTCOUNTS2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HISTCOUNTS2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HISTCOUNTS2_ERRORS,
};
