use crate::*;

const HISTCOUNTS_OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Histogram bin counts.",
}];

const HISTCOUNTS_OUTPUT_NE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "edges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin edge vector.",
    },
];

const HISTCOUNTS_OUTPUT_NEB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "edges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin edge vector.",
    },
    BuiltinParamDescriptor {
        name: "bin",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bin indices in the shape of X; zero for excluded observations.",
    },
];

const HISTCOUNTS_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data values.",
}];

const HISTCOUNTS_INPUTS_X_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data values.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar NumBins or explicit edge vector.",
    },
];

const HISTCOUNTS_INPUTS_X_NAMEVALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data values.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Name/value pairs (BinEdges, NumBins, BinWidth, BinLimits, Normalization, BinMethod).",
    },
];

const HISTCOUNTS_INPUTS_X_BINS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input data values.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar NumBins or explicit edge vector.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional name/value pairs.",
    },
];

const HISTCOUNTS_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "N = histcounts(X)",
        inputs: &HISTCOUNTS_INPUTS_X,
        outputs: &HISTCOUNTS_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts(X, bins)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS,
        outputs: &HISTCOUNTS_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts(X, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts(X, bins, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges] = histcounts(X)",
        inputs: &HISTCOUNTS_INPUTS_X,
        outputs: &HISTCOUNTS_OUTPUT_NE,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges] = histcounts(X, bins)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS,
        outputs: &HISTCOUNTS_OUTPUT_NE,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges] = histcounts(X, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_NE,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges] = histcounts(X, bins, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_NE,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges, bin] = histcounts(X)",
        inputs: &HISTCOUNTS_INPUTS_X,
        outputs: &HISTCOUNTS_OUTPUT_NEB,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges, bin] = histcounts(X, bins)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS,
        outputs: &HISTCOUNTS_OUTPUT_NEB,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges, bin] = histcounts(X, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_NEB,
    },
    BuiltinSignatureDescriptor {
        label: "[N, edges, bin] = histcounts(X, bins, Name, Value, ...)",
        inputs: &HISTCOUNTS_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HISTCOUNTS_OUTPUT_NEB,
    },
];

pub const HISTCOUNTS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS.INVALID_ARGUMENT",
    identifier: Some("RunMat:histcounts:InvalidArgument"),
    when: "Arguments are malformed, inconsistent, or unsupported.",
    message: "histcounts: invalid argument",
};

pub const HISTCOUNTS_ERROR_BINMETHOD_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS.BINMETHOD_CONFLICT",
    identifier: Some("RunMat:histcounts:BinMethodConflict"),
    when: "BinMethod is combined with incompatible bin controls.",
    message: "histcounts: BinMethod cannot be combined with BinEdges, NumBins, or BinWidth",
};

pub const HISTCOUNTS_ERROR_BINWIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS.BINWIDTH_INVALID",
    identifier: Some("RunMat:histcounts:BinWidthInvalid"),
    when: "BinWidth is zero, negative, non-finite, or non-scalar.",
    message: "histcounts: BinWidth must be a positive finite scalar",
};

pub const HISTCOUNTS_ERROR_EDGE_PRECISION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS.EDGE_PRECISION",
    identifier: Some("RunMat:histcounts:EdgePrecision"),
    when: "Integer-aligned unit bin boundaries cannot be represented distinctly by the double edge output.",
    message: "histcounts: integer-aligned bin boundaries cannot be represented distinctly as double",
};

pub const HISTCOUNTS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS.INTERNAL",
    identifier: Some("RunMat:histcounts:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "histcounts: internal operation failed",
};

const HISTCOUNTS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    HISTCOUNTS_ERROR_INVALID_ARGUMENT,
    HISTCOUNTS_ERROR_BINMETHOD_CONFLICT,
    HISTCOUNTS_ERROR_BINWIDTH_INVALID,
    HISTCOUNTS_ERROR_EDGE_PRECISION,
    HISTCOUNTS_ERROR_INTERNAL,
];

pub const HISTCOUNTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HISTCOUNTS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HISTCOUNTS_ERRORS,
};
