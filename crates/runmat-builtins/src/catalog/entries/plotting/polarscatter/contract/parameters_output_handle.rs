use crate::*;

pub(super) const POLARSCATTER_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "ps",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scatter handle or row-vector of scatter handles.",
    }];

pub(super) const PARAM_THETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular coordinates in radians.",
};

pub(super) const PARAM_RHO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rho",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Radial coordinates.",
};

pub(super) const PARAM_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("36"),
    description: "Marker area in points squared, as scalar, vector, or matrix.",
};

pub(super) const PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Marker color as color name, RGB triplet, scalar vector, or RGB matrix.",
};

pub(super) const PARAM_ARGS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional marker, filled flag, axes handle, and name/value properties.",
};

pub(super) const INPUT_THETA_RHO: [BuiltinParamDescriptor; 2] = [PARAM_THETA, PARAM_RHO];

pub(super) const INPUT_THETA_RHO_SZ: [BuiltinParamDescriptor; 3] =
    [PARAM_THETA, PARAM_RHO, PARAM_SZ];

pub(super) const INPUT_THETA_RHO_SZ_C: [BuiltinParamDescriptor; 4] =
    [PARAM_THETA, PARAM_RHO, PARAM_SZ, PARAM_C];

pub(super) const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [PARAM_ARGS];
