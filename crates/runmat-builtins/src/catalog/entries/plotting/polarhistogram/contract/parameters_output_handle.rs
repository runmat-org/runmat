use crate::*;

pub(super) const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created polar histogram chart.",
}];

pub(super) const INPUT_THETA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular sample data in radians.",
}];

pub(super) const INPUT_THETA_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "theta",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Angular sample data in radians.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector in radians.",
    },
];

pub(super) const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional axes handle, bin specification, and name/value pairs.",
}];

pub(super) const POLARHISTOGRAM_INPUTS_MANUAL: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "BinEdges/BinCounts pairs and optional style properties.",
    }];
