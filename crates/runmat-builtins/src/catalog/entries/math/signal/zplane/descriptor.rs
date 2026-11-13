use crate::*;
const ZPLANE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics handles for zeros, poles, unit circle, and axes guides.",
}];

const ZPLANE_INPUTS_BA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector or zero locations.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector or pole locations.",
    },
];

const ZPLANE_INPUTS_ZPK: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero locations.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pole locations.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "System gain; accepted for zpk compatibility and ignored for plotting.",
    },
];

const ZPLANE_INPUTS_SOS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sos",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second-order-section matrix with six columns [b0 b1 b2 a0 a1 a2].",
}];

const ZPLANE_INPUTS_AX_SOS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes.",
    },
    BuiltinParamDescriptor {
        name: "sos",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second-order-section matrix with six columns.",
    },
];
const ZPLANE_INPUTS_AX_BA: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector or zero locations.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector or pole locations.",
    },
];
const ZPLANE_INPUTS_AX_ZPK: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero locations.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pole locations.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite gain, ignored for marker placement.",
    },
];

const ZPLANE_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "zplane(b, a)",
        inputs: &ZPLANE_INPUTS_BA,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(z, p, k)",
        inputs: &ZPLANE_INPUTS_ZPK,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(sos)",
        inputs: &ZPLANE_INPUTS_SOS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(b, a)",
        inputs: &ZPLANE_INPUTS_BA,
        outputs: &ZPLANE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(z, p, k)",
        inputs: &ZPLANE_INPUTS_ZPK,
        outputs: &ZPLANE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(sos)",
        inputs: &ZPLANE_INPUTS_SOS,
        outputs: &ZPLANE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "zplane(ax, sos)",
        inputs: &ZPLANE_INPUTS_AX_SOS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(ax, b, a)",
        inputs: &ZPLANE_INPUTS_AX_BA,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "zplane(ax, z, p, k)",
        inputs: &ZPLANE_INPUTS_AX_ZPK,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(ax, sos)",
        inputs: &ZPLANE_INPUTS_AX_SOS,
        outputs: &ZPLANE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(ax, b, a)",
        inputs: &ZPLANE_INPUTS_AX_BA,
        outputs: &ZPLANE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = zplane(ax, z, p, k)",
        inputs: &ZPLANE_INPUTS_AX_ZPK,
        outputs: &ZPLANE_OUTPUT,
    },
];

pub const ZPLANE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZPLANE.INVALID_ARGUMENT",
    identifier: Some("RunMat:zplane:InvalidArgument"),
    when: "Inputs cannot be interpreted as filter coefficients, zero/pole vectors, or SOS rows.",
    message: "zplane: invalid argument",
};

pub const ZPLANE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZPLANE.INTERNAL",
    identifier: Some("RunMat:zplane:Internal"),
    when: "Internal plotting state or tensor construction fails.",
    message: "zplane: internal operation failed",
};

const ZPLANE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ZPLANE_ERROR_INVALID_ARGUMENT, ZPLANE_ERROR_INTERNAL];

pub const ZPLANE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZPLANE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZPLANE_ERRORS,
};
