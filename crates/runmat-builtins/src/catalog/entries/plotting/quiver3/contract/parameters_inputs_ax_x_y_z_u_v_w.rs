use crate::*;

pub(super) const INPUTS_AX_X_Y_Z_U_V_W: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
];

pub(super) const INPUTS_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];

pub(super) const INPUTS_X_Y_Z_U_V_W_ARGS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base x-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base y-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Arrow base z-coordinates.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field x-components.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field y-components.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector field z-components.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional scale, line-style shorthand, or name/value style arguments.",
    },
];
