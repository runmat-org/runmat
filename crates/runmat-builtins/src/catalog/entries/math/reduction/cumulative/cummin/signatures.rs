use crate::*;
const CUMMIN_OUTPUT_M: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cumulative minimum values.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based double indices along the scan dimension.",
    },
];

const CUMMIN_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real or complex single/double, real integer, or logical scalar or N-D array; sparse, distributed and complex-integer inputs are unsupported.",
};

const CUMMIN_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Positive finite integer-valued numeric scalar dimension, or [] to use the first nonsingleton dimension. No dimension vector is accepted.",
};

const CUMMIN_PARAM_DIRECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "direction",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"forward\""),
    description: "Scan direction: \"forward\" or \"reverse\".",
};

const CUMMIN_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description:
        "Missing-value mode: \"includenan\"/\"includemissing\" or \"omitnan\"/\"omitmissing\".",
};

const CUMMIN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [CUMMIN_PARAM_A];
const CUMMIN_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [CUMMIN_PARAM_A, CUMMIN_PARAM_DIM];
const CUMMIN_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_DIRECTION];
const CUMMIN_INPUTS_NANFLAG: [BuiltinParamDescriptor; 2] = [CUMMIN_PARAM_A, CUMMIN_PARAM_NANFLAG];
const CUMMIN_INPUTS_DIM_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_DIM, CUMMIN_PARAM_DIRECTION];
const CUMMIN_INPUTS_DIRECTION_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_DIRECTION, CUMMIN_PARAM_DIM];
const CUMMIN_INPUTS_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_DIM, CUMMIN_PARAM_NANFLAG];
const CUMMIN_INPUTS_NANFLAG_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_NANFLAG, CUMMIN_PARAM_DIM];
const CUMMIN_INPUTS_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_DIRECTION, CUMMIN_PARAM_NANFLAG];
const CUMMIN_INPUTS_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMIN_PARAM_A, CUMMIN_PARAM_NANFLAG, CUMMIN_PARAM_DIRECTION];
const CUMMIN_INPUTS_DIM_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_DIM,
    CUMMIN_PARAM_DIRECTION,
    CUMMIN_PARAM_NANFLAG,
];
const CUMMIN_INPUTS_DIM_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_DIM,
    CUMMIN_PARAM_NANFLAG,
    CUMMIN_PARAM_DIRECTION,
];
const CUMMIN_INPUTS_DIRECTION_DIM_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_DIRECTION,
    CUMMIN_PARAM_DIM,
    CUMMIN_PARAM_NANFLAG,
];
const CUMMIN_INPUTS_DIRECTION_NANFLAG_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_DIRECTION,
    CUMMIN_PARAM_NANFLAG,
    CUMMIN_PARAM_DIM,
];
const CUMMIN_INPUTS_NANFLAG_DIM_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_NANFLAG,
    CUMMIN_PARAM_DIM,
    CUMMIN_PARAM_DIRECTION,
];
const CUMMIN_INPUTS_NANFLAG_DIRECTION_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMIN_PARAM_A,
    CUMMIN_PARAM_NANFLAG,
    CUMMIN_PARAM_DIRECTION,
    CUMMIN_PARAM_DIM,
];

pub(super) const CUMMIN_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A)",
        inputs: &CUMMIN_INPUTS_CORE,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, dim)",
        inputs: &CUMMIN_INPUTS_DIM,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, direction)",
        inputs: &CUMMIN_INPUTS_DIRECTION,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, nanflag)",
        inputs: &CUMMIN_INPUTS_NANFLAG,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, dim, direction)",
        inputs: &CUMMIN_INPUTS_DIM_DIRECTION,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, direction, dim)",
        inputs: &CUMMIN_INPUTS_DIRECTION_DIM,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, dim, nanflag)",
        inputs: &CUMMIN_INPUTS_DIM_NANFLAG,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, nanflag, dim)",
        inputs: &CUMMIN_INPUTS_NANFLAG_DIM,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, direction, nanflag)",
        inputs: &CUMMIN_INPUTS_DIRECTION_NANFLAG,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, nanflag, direction)",
        inputs: &CUMMIN_INPUTS_NANFLAG_DIRECTION,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, dim, direction, nanflag)",
        inputs: &CUMMIN_INPUTS_DIM_DIRECTION_NANFLAG,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, dim, nanflag, direction)",
        inputs: &CUMMIN_INPUTS_DIM_NANFLAG_DIRECTION,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, direction, dim, nanflag)",
        inputs: &CUMMIN_INPUTS_DIRECTION_DIM_NANFLAG,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, direction, nanflag, dim)",
        inputs: &CUMMIN_INPUTS_DIRECTION_NANFLAG_DIM,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, nanflag, dim, direction)",
        inputs: &CUMMIN_INPUTS_NANFLAG_DIM_DIRECTION,
        outputs: &CUMMIN_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "[M, I] = cummin(A, nanflag, direction, dim)",
        inputs: &CUMMIN_INPUTS_NANFLAG_DIRECTION_DIM,
        outputs: &CUMMIN_OUTPUT_M,
    },
];
