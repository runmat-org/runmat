use crate::*;

pub(super) const TRIPLOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the line object containing the triangulation edges.",
}];

pub(super) const PARAM_TR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "TR",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Triangulation-like struct with ConnectivityList and Points fields.",
};

pub(super) const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

pub(super) const PARAM_TRI: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "TRI",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "M-by-3 triangle connectivity list using MATLAB one-based vertex indices.",
};

pub(super) const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vertex X coordinates.",
};

pub(super) const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vertex Y coordinates.",
};

pub(super) const PARAM_LINE_SPEC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lineSpec",
    ty: BuiltinParamType::StyleSpec,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Line style, marker, and color shorthand.",
};

pub(super) const PARAM_PROPS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name/value line style properties such as Color, LineWidth, and DisplayName.",
};

pub(super) const INPUTS_TRI_X_Y: [BuiltinParamDescriptor; 3] = [PARAM_TRI, PARAM_X, PARAM_Y];

pub(super) const INPUTS_TRI_X_Y_STYLE: [BuiltinParamDescriptor; 4] =
    [PARAM_TRI, PARAM_X, PARAM_Y, PARAM_LINE_SPEC];

pub(super) const INPUTS_TRI_X_Y_PROPS: [BuiltinParamDescriptor; 4] =
    [PARAM_TRI, PARAM_X, PARAM_Y, PARAM_PROPS];

pub(super) const INPUTS_AX_TRI_X_Y: [BuiltinParamDescriptor; 4] =
    [PARAM_AX, PARAM_TRI, PARAM_X, PARAM_Y];

pub(super) const INPUTS_TR: [BuiltinParamDescriptor; 1] = [PARAM_TR];

pub(super) const INPUTS_TR_STYLE: [BuiltinParamDescriptor; 2] = [PARAM_TR, PARAM_LINE_SPEC];

pub(super) const INPUTS_AX_TR_STYLE: [BuiltinParamDescriptor; 3] =
    [PARAM_AX, PARAM_TR, PARAM_LINE_SPEC];
