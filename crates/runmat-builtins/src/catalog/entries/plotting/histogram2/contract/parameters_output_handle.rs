use crate::*;

pub(super) const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created histogram2 chart.",
}];

pub(super) const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-axis input samples.",
};

pub(super) const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y-axis input samples.",
};

pub(super) const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

pub(super) const INPUTS_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];

pub(super) const INPUTS_XY_BINS: [BuiltinParamDescriptor; 3] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar/two-element bin count or edge specification.",
    },
];

pub(super) const INPUTS_XY_BINXY: [BuiltinParamDescriptor; 4] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin count scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin count scalar or edge vector.",
    },
];

pub(super) const INPUTS_XY_OPTIONS: [BuiltinParamDescriptor; 3] = [
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Binning and chart name/value options.",
    },
];

pub(super) const INPUTS_AX_XY: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_X, PARAM_Y];

pub(super) const INPUTS_AX_XY_BINS: [BuiltinParamDescriptor; 4] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar/two-element bin count or edge specification.",
    },
];

pub(super) const INPUTS_AX_XY_BINXY: [BuiltinParamDescriptor; 5] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin count scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin count scalar or edge vector.",
    },
];

pub(super) const INPUTS_AX_XY_OPTIONS: [BuiltinParamDescriptor; 4] = [
    PARAM_AX,
    PARAM_X,
    PARAM_Y,
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Binning and chart name/value options.",
    },
];
