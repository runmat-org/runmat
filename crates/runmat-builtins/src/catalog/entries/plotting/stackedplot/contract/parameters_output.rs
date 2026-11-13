use crate::*;

pub(super) const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the StackedLineChart object.",
}];

pub(super) const INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector or matrix data. Matrix columns are plotted in separate stacked axes.",
}];

pub(super) const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Common x-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector or matrix y data.",
    },
];

pub(super) const INPUTS_TABLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table or timetable data.",
}];

pub(super) const INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable selectors, line style, and StackedLineChart name-value options.",
}];
