use crate::*;

pub(super) const FSURF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered function surface.",
}];

pub(super) const FSURF_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function of two variables, z = f(x,y).",
}];

pub(super) const FSURF_INPUTS_F_DOMAIN: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function of two variables, z = f(x,y).",
    },
    BuiltinParamDescriptor {
        name: "xyinterval",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[-5 5 -5 5]"),
        description: "Two- or four-element domain vector.",
    },
];

pub(super) const FSURF_INPUTS_PARAMETRIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric x-coordinate function.",
    },
    BuiltinParamDescriptor {
        name: "fy",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric y-coordinate function.",
    },
    BuiltinParamDescriptor {
        name: "fz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parametric z-coordinate function.",
    },
];

pub(super) const FSURF_INPUTS_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and surface style name/value arguments.",
    },
];

pub(super) const FSURF_INPUTS_AX_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function-handle and optional domain arguments.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "MeshDensity and surface style name/value arguments.",
    },
];
