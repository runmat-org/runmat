use crate::*;
const CUMMAX_OUTPUT_M: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cumulative maximum values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based double indices along the scan dimension.",
    },
];

const CUMMAX_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real or complex single/double, real integer, or logical scalar or N-D array; sparse, distributed and complex-integer inputs are unsupported.",
};

const CUMMAX_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Positive finite integer-valued numeric scalar dimension, or [] to use the first nonsingleton dimension. No dimension vector is accepted.",
};

const CUMMAX_PARAM_DIRECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "direction",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"forward\""),
    description: "Scan direction: \"forward\" or \"reverse\".",
};

const CUMMAX_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description:
        "Missing-value mode: \"includenan\"/\"includemissing\" or \"omitnan\"/\"omitmissing\".",
};

const CUMMAX_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [CUMMAX_PARAM_A];
const CUMMAX_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_NANFLAG: [BuiltinParamDescriptor; 2] = [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_DIM_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_DIRECTION_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_NANFLAG_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_DIM_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_NANFLAG,
];
const CUMMAX_INPUTS_DIM_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIRECTION,
];
const CUMMAX_INPUTS_DIRECTION_DIM_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_NANFLAG,
];
const CUMMAX_INPUTS_DIRECTION_NANFLAG_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIM,
];
const CUMMAX_INPUTS_NANFLAG_DIM_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_DIRECTION,
];
const CUMMAX_INPUTS_NANFLAG_DIRECTION_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_DIM,
];

pub(super) const CUMMAX_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A)",
        inputs: &CUMMAX_INPUTS_CORE,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, dim)",
        inputs: &CUMMAX_INPUTS_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, direction)",
        inputs: &CUMMAX_INPUTS_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, nanflag)",
        inputs: &CUMMAX_INPUTS_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, dim, direction)",
        inputs: &CUMMAX_INPUTS_DIM_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, direction, dim)",
        inputs: &CUMMAX_INPUTS_DIRECTION_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, dim, nanflag)",
        inputs: &CUMMAX_INPUTS_DIM_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, nanflag, dim)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, direction, nanflag)",
        inputs: &CUMMAX_INPUTS_DIRECTION_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, nanflag, direction)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, dim, direction, nanflag)",
        inputs: &CUMMAX_INPUTS_DIM_DIRECTION_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, dim, nanflag, direction)",
        inputs: &CUMMAX_INPUTS_DIM_NANFLAG_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, direction, dim, nanflag)",
        inputs: &CUMMAX_INPUTS_DIRECTION_DIM_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, direction, nanflag, dim)",
        inputs: &CUMMAX_INPUTS_DIRECTION_NANFLAG_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, nanflag, dim, direction)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIM_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummax(A, nanflag, direction, dim)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIRECTION_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
];
