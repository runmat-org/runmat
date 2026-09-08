use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "names",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column cell array of character-row names.",
}];
const INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure, represented structure array, or supported RunMat object.",
}];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "names = fieldnames(S)",
    inputs: INPUT,
    outputs: OUTPUT,
}];

pub const FIELDNAMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
