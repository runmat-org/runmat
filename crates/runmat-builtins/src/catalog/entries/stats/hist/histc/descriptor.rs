use crate::*;
/// Allocation/rank admission limit of the legacy histogram implementation.
pub const HISTC_MAX_DIMENSION: usize = 32;
const OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bincounts",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy histogram bin counts.",
}];

const OUTPUT_N_IND: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bincounts",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Legacy histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "ind",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bin index for each input element, or zero when outside all bins.",
    },
];

const INPUT_X_EDGES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input values to sort into legacy bins.",
    },
    BuiltinParamDescriptor {
        name: "binranges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Monotonically nondecreasing bin endpoints.",
    },
];

const INPUT_X_EDGES_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input values to sort into legacy bins.",
    },
    BuiltinParamDescriptor {
        name: "binranges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Monotonically nondecreasing bin endpoints.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension along which to operate.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "bincounts = histc(x, binranges)",
        inputs: &INPUT_X_EDGES,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "bincounts = histc(x, binranges, dim)",
        inputs: &INPUT_X_EDGES_DIM,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[bincounts, ind] = histc(x, binranges)",
        inputs: &INPUT_X_EDGES,
        outputs: &OUTPUT_N_IND,
    },
    BuiltinSignatureDescriptor {
        label: "[bincounts, ind] = histc(x, binranges, dim)",
        inputs: &INPUT_X_EDGES_DIM,
        outputs: &OUTPUT_N_IND,
    },
];

pub const HISTC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTC.INVALID_ARGUMENT",
    identifier: Some("RunMat:histc:InvalidArgument"),
    when: "Arguments are malformed, inconsistent, or unsupported.",
    message: "histc: invalid argument",
};

pub const HISTC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTC.INTERNAL",
    identifier: Some("RunMat:histc:Internal"),
    when: "Internal tensor conversion, GPU gather, or allocation fails.",
    message: "histc: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [HISTC_ERROR_INVALID_ARGUMENT, HISTC_ERROR_INTERNAL];

pub const HISTC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
