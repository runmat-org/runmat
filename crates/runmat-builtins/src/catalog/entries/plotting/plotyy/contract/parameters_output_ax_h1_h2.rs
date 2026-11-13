use crate::*;

pub(super) const OUTPUT_AX_H1_H2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two axes handles [left right].",
    },
    BuiltinParamDescriptor {
        name: "h1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line handle or handles for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "h2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line handle or handles for the right axes.",
    },
];

pub(super) const INPUTS_X1_Y1_X2_Y2: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "Y1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data for the left axes.",
    },
    BuiltinParamDescriptor {
        name: "X2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data for the right axes.",
    },
    BuiltinParamDescriptor {
        name: "Y2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data for the right axes.",
    },
];

pub(super) const INPUTS_WITH_FUN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "X1, Y1, X2, Y2 followed by one plotting function selector.",
}];

pub(super) const INPUTS_WITH_FUNS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "X1, Y1, X2, Y2 followed by left and right plotting function selectors.",
}];
