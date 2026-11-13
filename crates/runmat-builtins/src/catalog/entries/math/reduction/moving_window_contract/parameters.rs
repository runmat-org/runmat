use crate::*;

pub(in crate::catalog::entries::math::reduction) const OUTPUT_Y: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Moving-window reduction result.",
    }];

pub(in crate::catalog::entries::math::reduction) const PARAM_A: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input numeric, logical, or gpuArray data.",
    };

pub(in crate::catalog::entries::math::reduction) const PARAM_K: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Window length or two-element [kb kf] backward/forward window.",
    };

pub(in crate::catalog::entries::math::reduction) const PARAM_DIM: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dimension along which to move the window.",
    };

pub(in crate::catalog::entries::math::reduction) const PARAM_W: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description:
            "Normalization flag: 0 for sample normalization, 1 for population normalization.",
    };

pub(in crate::catalog::entries::math::reduction) const PARAM_NANFLAG: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "nanflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includenan\""),
        description: "NaN handling mode: \"includenan\" or \"omitnan\".",
    };

pub(in crate::catalog::entries::math::reduction) const PARAM_OPTIONS: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options, including \"Endpoints\".",
    };

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K: [BuiltinParamDescriptor; 2] =
    [PARAM_A, PARAM_K];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_DIM: [BuiltinParamDescriptor; 3] =
    [PARAM_A, PARAM_K, PARAM_DIM];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_NANFLAG:
    [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_K, PARAM_NANFLAG];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_NV: [BuiltinParamDescriptor; 3] =
    [PARAM_A, PARAM_K, PARAM_OPTIONS];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_OPTIONS:
    [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_K, PARAM_DIM, PARAM_OPTIONS];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W: [BuiltinParamDescriptor; 3] =
    [PARAM_A, PARAM_K, PARAM_W];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_DIM: [BuiltinParamDescriptor;
    4] = [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_NANFLAG:
    [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_K, PARAM_W, PARAM_NANFLAG];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_NANFLAG_NV:
    [BuiltinParamDescriptor; 5] = [PARAM_A, PARAM_K, PARAM_W, PARAM_NANFLAG, PARAM_OPTIONS];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_DIM_NANFLAG:
    [BuiltinParamDescriptor; 5] = [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM, PARAM_NANFLAG];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_NV: [BuiltinParamDescriptor;
    4] = [PARAM_A, PARAM_K, PARAM_W, PARAM_OPTIONS];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_DIM_NV:
    [BuiltinParamDescriptor; 5] = [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM, PARAM_OPTIONS];

pub(in crate::catalog::entries::math::reduction) const INPUTS_A_K_W_DIM_NANFLAG_NV:
    [BuiltinParamDescriptor; 6] = [
    PARAM_A,
    PARAM_K,
    PARAM_W,
    PARAM_DIM,
    PARAM_NANFLAG,
    PARAM_OPTIONS,
];
