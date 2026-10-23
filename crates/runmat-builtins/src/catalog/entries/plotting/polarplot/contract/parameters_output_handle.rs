use crate::*;

pub(super) const POLARPLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the first polar line object.",
}];

pub(super) const POLARPLOT_INPUTS_THETA_RHO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "theta",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Angular coordinates in radians.",
    },
    BuiltinParamDescriptor {
        name: "rho",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Radial coordinates.",
    },
];

pub(super) const POLARPLOT_INPUTS_RHO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "rhoOrZ",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Radial data with implicit theta, or complex data interpreted as angle/abs.",
}];

pub(super) const POLARPLOT_INPUTS_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Polar plot inputs with optional axes handle, LineSpec, and Name/Value pairs.",
}];
