use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const ORDERED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure with reordered top-level fields.",
};
const PERMUTATION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Pout",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double column vector mapping output fields to their original positions.",
};
const TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input structure.",
};
const ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "order",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reference structure, field-name collection, or numeric permutation vector.",
};
const ONE_OUTPUT: &[BuiltinParamDescriptor] = &[ORDERED];
const TWO_OUTPUTS: &[BuiltinParamDescriptor] = &[ORDERED, PERMUTATION];
const TARGET_ONLY: &[BuiltinParamDescriptor] = &[TARGET];
const TARGET_AND_ORDER: &[BuiltinParamDescriptor] = &[TARGET, ORDER];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "S = orderfields(S1)",
        inputs: TARGET_ONLY,
        outputs: ONE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = orderfields(S1, order)",
        inputs: TARGET_AND_ORDER,
        outputs: ONE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[S, Pout] = orderfields(S1)",
        inputs: TARGET_ONLY,
        outputs: TWO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, Pout] = orderfields(S1, order)",
        inputs: TARGET_AND_ORDER,
        outputs: TWO_OUTPUTS,
    },
];

pub const ORDERFIELDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
