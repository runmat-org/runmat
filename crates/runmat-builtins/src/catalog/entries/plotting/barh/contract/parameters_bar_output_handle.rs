use crate::*;

pub(super) const BAR_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bar graphics handle or handle row vector, with one handle per plotted series.",
}];

pub(super) const BAR_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bar heights as a vector or matrix.",
}];

pub(super) const BAR_INPUTS_Y_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "style",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Layout/style token such as 'stacked' or color shorthand.",
    },
];

pub(super) const BAR_INPUTS_Y_WIDTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.8"),
        description: "Fraction of available category space occupied by each bar.",
    },
];

pub(super) const BAR_INPUTS_Y_WIDTH_COLOR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.8"),
        description: "Fraction of available category space occupied by each bar.",
    },
    BuiltinParamDescriptor {
        name: "color",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Color shorthand or color name for all bars.",
    },
];

pub(super) const BAR_INPUTS_Y_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style arguments.",
    },
];

pub(super) const BAR_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
];

pub(super) const BAR_INPUTS_X_Y_STYLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "style",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Layout/style token such as 'stacked' or color shorthand.",
    },
];

pub(super) const BAR_INPUTS_X_Y_WIDTH: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Category positions for bars.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bar heights as a vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "width",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0.8"),
        description: "Fraction of available category space occupied by each bar.",
    },
];
