use crate::*;

pub(super) const PARAM_S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix of scatter graphics handles for off-diagonal plots.",
};

pub(super) const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "AX",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix of axes handles used by the plot matrix.",
};

pub(super) const PARAM_BIG_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BigAx",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Compatibility axes handle for labels spanning the plot matrix.",
};

pub(super) const PARAM_H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector of diagonal histogram handles for the one-input form.",
};

pub(super) const PARAM_H_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "HAx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector of diagonal histogram axes handles for the one-input form.",
};

pub(super) const OUTPUT_S: [BuiltinParamDescriptor; 1] = [PARAM_S];

pub(super) const OUTPUT_S_AX: [BuiltinParamDescriptor; 2] = [PARAM_S, PARAM_AX];

pub(super) const OUTPUT_S_AX_BIGAX: [BuiltinParamDescriptor; 3] = [PARAM_S, PARAM_AX, PARAM_BIG_AX];

pub(super) const OUTPUT_MULTI: [BuiltinParamDescriptor; 5] =
    [PARAM_S, PARAM_AX, PARAM_BIG_AX, PARAM_H, PARAM_H_AX];

pub(super) const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix whose columns are variables and rows are observations.",
};

pub(super) const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix whose columns are variables and rows are observations.",
};

pub(super) const PARAM_LINE_SPEC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lineSpec",
    ty: BuiltinParamType::StyleSpec,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional marker/color style applied to scatter plots.",
};

pub(super) const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];

pub(super) const INPUT_X_LINE_SPEC: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_LINE_SPEC];

pub(super) const INPUT_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];

pub(super) const INPUT_LINE_SPEC: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_LINE_SPEC];
