use crate::*;
const CUMPROD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative product result.",
}];

const CUMPROD_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real or complex single/double, real integer, or logical scalar or N-D array; sparse, distributed and complex-integer inputs are unsupported.",
};

const CUMPROD_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Positive finite integer-valued numeric scalar dimension, or [] to use the first nonsingleton dimension. No dimension vector is accepted.",
};

const CUMPROD_PARAM_DIRECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "direction",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"forward\""),
    description: "Scan direction: \"forward\" or \"reverse\".",
};

const CUMPROD_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description:
        "Missing-value mode: \"includenan\"/\"includemissing\" or \"omitnan\"/\"omitmissing\".",
};

const CUMPROD_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [CUMPROD_PARAM_A];

const CUMPROD_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [CUMPROD_PARAM_A, CUMPROD_PARAM_DIM];

const CUMPROD_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_DIRECTION];

const CUMPROD_INPUTS_NANFLAG: [BuiltinParamDescriptor; 2] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_NANFLAG];

const CUMPROD_INPUTS_DIM_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_DIM, CUMPROD_PARAM_DIRECTION];

const CUMPROD_INPUTS_DIRECTION_DIM: [BuiltinParamDescriptor; 3] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_DIRECTION, CUMPROD_PARAM_DIM];

const CUMPROD_INPUTS_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_DIM, CUMPROD_PARAM_NANFLAG];

const CUMPROD_INPUTS_NANFLAG_DIM: [BuiltinParamDescriptor; 3] =
    [CUMPROD_PARAM_A, CUMPROD_PARAM_NANFLAG, CUMPROD_PARAM_DIM];

const CUMPROD_INPUTS_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 3] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_DIRECTION,
    CUMPROD_PARAM_NANFLAG,
];

const CUMPROD_INPUTS_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 3] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_NANFLAG,
    CUMPROD_PARAM_DIRECTION,
];

const CUMPROD_INPUTS_DIM_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_DIM,
    CUMPROD_PARAM_DIRECTION,
    CUMPROD_PARAM_NANFLAG,
];

const CUMPROD_INPUTS_DIM_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_DIM,
    CUMPROD_PARAM_NANFLAG,
    CUMPROD_PARAM_DIRECTION,
];

const CUMPROD_INPUTS_DIRECTION_DIM_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_DIRECTION,
    CUMPROD_PARAM_DIM,
    CUMPROD_PARAM_NANFLAG,
];

const CUMPROD_INPUTS_DIRECTION_NANFLAG_DIM: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_DIRECTION,
    CUMPROD_PARAM_NANFLAG,
    CUMPROD_PARAM_DIM,
];

const CUMPROD_INPUTS_NANFLAG_DIM_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_NANFLAG,
    CUMPROD_PARAM_DIM,
    CUMPROD_PARAM_DIRECTION,
];

const CUMPROD_INPUTS_NANFLAG_DIRECTION_DIM: [BuiltinParamDescriptor; 4] = [
    CUMPROD_PARAM_A,
    CUMPROD_PARAM_NANFLAG,
    CUMPROD_PARAM_DIRECTION,
    CUMPROD_PARAM_DIM,
];

pub(super) const CUMPROD_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A)",
        inputs: &CUMPROD_INPUTS_CORE,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, dim)",
        inputs: &CUMPROD_INPUTS_DIM,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, direction)",
        inputs: &CUMPROD_INPUTS_DIRECTION,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, nanflag)",
        inputs: &CUMPROD_INPUTS_NANFLAG,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, dim, direction)",
        inputs: &CUMPROD_INPUTS_DIM_DIRECTION,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, direction, dim)",
        inputs: &CUMPROD_INPUTS_DIRECTION_DIM,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, dim, nanflag)",
        inputs: &CUMPROD_INPUTS_DIM_NANFLAG,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, nanflag, dim)",
        inputs: &CUMPROD_INPUTS_NANFLAG_DIM,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, direction, nanflag)",
        inputs: &CUMPROD_INPUTS_DIRECTION_NANFLAG,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, nanflag, direction)",
        inputs: &CUMPROD_INPUTS_NANFLAG_DIRECTION,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, dim, direction, nanflag)",
        inputs: &CUMPROD_INPUTS_DIM_DIRECTION_NANFLAG,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, dim, nanflag, direction)",
        inputs: &CUMPROD_INPUTS_DIM_NANFLAG_DIRECTION,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, direction, dim, nanflag)",
        inputs: &CUMPROD_INPUTS_DIRECTION_DIM_NANFLAG,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, direction, nanflag, dim)",
        inputs: &CUMPROD_INPUTS_DIRECTION_NANFLAG_DIM,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, nanflag, dim, direction)",
        inputs: &CUMPROD_INPUTS_NANFLAG_DIM_DIRECTION,
        outputs: &CUMPROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cumprod(A, nanflag, direction, dim)",
        inputs: &CUMPROD_INPUTS_NANFLAG_DIRECTION_DIM,
        outputs: &CUMPROD_OUTPUT,
    },
];
