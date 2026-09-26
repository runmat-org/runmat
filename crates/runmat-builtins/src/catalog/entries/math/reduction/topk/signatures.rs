use crate::*;
pub(super) const VALUE_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected values.",
};

pub(super) const INDEX_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based indices along the selected dimension.",
};

pub(super) const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [VALUE_OUTPUT];
pub(super) const OUTPUT_VALUE_INDEX: [BuiltinParamDescriptor; 2] = [VALUE_OUTPUT, INDEX_OUTPUT];

pub(super) const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};

pub(super) const INPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements to select from each slice.",
};

pub(super) const INPUT_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension to operate along.",
};

pub(super) const INPUT_OPTION_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"ComparisonMethod\""),
    description: "Name-value option name.",
};

pub(super) const INPUT_OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionValue",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"auto\""),
    description: "Name-value option value.",
};

pub(super) const INPUTS_A_K: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_K];
pub(super) const INPUTS_A_K_DIM: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_K, INPUT_DIM];
pub(super) const INPUTS_A_K_OPTIONS: [BuiltinParamDescriptor; 4] =
    [INPUT_A, INPUT_K, INPUT_OPTION_NAME, INPUT_OPTION_VALUE];
pub(super) const INPUTS_A_K_DIM_OPTIONS: [BuiltinParamDescriptor; 5] = [
    INPUT_A,
    INPUT_K,
    INPUT_DIM,
    INPUT_OPTION_NAME,
    INPUT_OPTION_VALUE,
];
