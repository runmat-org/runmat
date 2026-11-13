use crate::*;
const CUMSUM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative sum result.",
}];

const CUMSUM_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real or complex single/double, real integer, or logical scalar or N-D array; sparse, distributed and complex-integer inputs are unsupported.",
};

const CUMSUM_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Positive finite integer-valued numeric scalar dimension, or [] to use the first nonsingleton dimension. No dimension vector is accepted.",
};

const CUMSUM_PARAM_DIRECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "direction",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"forward\""),
    description: "Scan direction: \"forward\" or \"reverse\".",
};

const CUMSUM_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description:
        "Missing-value mode: \"includenan\"/\"includemissing\" or \"omitnan\"/\"omitmissing\".",
};

const CUMSUM_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [CUMSUM_PARAM_A];
const CUMSUM_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [CUMSUM_PARAM_A, CUMSUM_PARAM_DIM];
const CUMSUM_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_DIRECTION];
const CUMSUM_INPUTS_NANFLAG: [BuiltinParamDescriptor; 2] = [CUMSUM_PARAM_A, CUMSUM_PARAM_NANFLAG];
const CUMSUM_INPUTS_DIM_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_DIM, CUMSUM_PARAM_DIRECTION];
const CUMSUM_INPUTS_DIRECTION_DIM: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_DIRECTION, CUMSUM_PARAM_DIM];
const CUMSUM_INPUTS_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_DIM, CUMSUM_PARAM_NANFLAG];
const CUMSUM_INPUTS_NANFLAG_DIM: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_NANFLAG, CUMSUM_PARAM_DIM];
const CUMSUM_INPUTS_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_DIRECTION, CUMSUM_PARAM_NANFLAG];
const CUMSUM_INPUTS_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMSUM_PARAM_A, CUMSUM_PARAM_NANFLAG, CUMSUM_PARAM_DIRECTION];
const CUMSUM_INPUTS_DIM_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_DIM,
    CUMSUM_PARAM_DIRECTION,
    CUMSUM_PARAM_NANFLAG,
];
const CUMSUM_INPUTS_DIM_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_DIM,
    CUMSUM_PARAM_NANFLAG,
    CUMSUM_PARAM_DIRECTION,
];
const CUMSUM_INPUTS_DIRECTION_DIM_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_DIRECTION,
    CUMSUM_PARAM_DIM,
    CUMSUM_PARAM_NANFLAG,
];
const CUMSUM_INPUTS_DIRECTION_NANFLAG_DIM: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_DIRECTION,
    CUMSUM_PARAM_NANFLAG,
    CUMSUM_PARAM_DIM,
];
const CUMSUM_INPUTS_NANFLAG_DIM_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_NANFLAG,
    CUMSUM_PARAM_DIM,
    CUMSUM_PARAM_DIRECTION,
];
const CUMSUM_INPUTS_NANFLAG_DIRECTION_DIM: [BuiltinParamDescriptor; 4] = [
    CUMSUM_PARAM_A,
    CUMSUM_PARAM_NANFLAG,
    CUMSUM_PARAM_DIRECTION,
    CUMSUM_PARAM_DIM,
];

pub(super) const CUMSUM_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A)",
        inputs: &CUMSUM_INPUTS_CORE,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, dim)",
        inputs: &CUMSUM_INPUTS_DIM,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, direction)",
        inputs: &CUMSUM_INPUTS_DIRECTION,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, nanflag)",
        inputs: &CUMSUM_INPUTS_NANFLAG,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, dim, direction)",
        inputs: &CUMSUM_INPUTS_DIM_DIRECTION,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, direction, dim)",
        inputs: &CUMSUM_INPUTS_DIRECTION_DIM,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, dim, nanflag)",
        inputs: &CUMSUM_INPUTS_DIM_NANFLAG,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, nanflag, dim)",
        inputs: &CUMSUM_INPUTS_NANFLAG_DIM,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, direction, nanflag)",
        inputs: &CUMSUM_INPUTS_DIRECTION_NANFLAG,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, nanflag, direction)",
        inputs: &CUMSUM_INPUTS_NANFLAG_DIRECTION,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, dim, direction, nanflag)",
        inputs: &CUMSUM_INPUTS_DIM_DIRECTION_NANFLAG,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, dim, nanflag, direction)",
        inputs: &CUMSUM_INPUTS_DIM_NANFLAG_DIRECTION,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, direction, dim, nanflag)",
        inputs: &CUMSUM_INPUTS_DIRECTION_DIM_NANFLAG,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, direction, nanflag, dim)",
        inputs: &CUMSUM_INPUTS_DIRECTION_NANFLAG_DIM,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, nanflag, dim, direction)",
        inputs: &CUMSUM_INPUTS_NANFLAG_DIM_DIRECTION,
        outputs: &CUMSUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumsum(A, nanflag, direction, dim)",
        inputs: &CUMSUM_INPUTS_NANFLAG_DIRECTION_DIM,
        outputs: &CUMSUM_OUTPUT,
    },
];
