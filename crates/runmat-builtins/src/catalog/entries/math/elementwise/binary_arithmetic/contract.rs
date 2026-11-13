use crate::*;

pub(super) const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise result with the broadcasted operand shape.",
};
pub(super) const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "First numeric, logical, character, symbolic, sparse, or provider-resident operand.",
};
pub(super) const INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second operand with a size compatible for implicit expansion.",
};
pub(super) const LIKE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "like",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Literal `like` selector for the RunMat output-prototype extension.",
};
pub(super) const PROTOTYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host, device, real, or complex output prototype.",
};

pub(super) const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[
    BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "An integer operand requires the other operand to have the same integer class or to be scalar double; complex integer arithmetic is rejected." },
    BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Compatible dimensions expand implicitly and the nondouble integer class is preserved." },
];
