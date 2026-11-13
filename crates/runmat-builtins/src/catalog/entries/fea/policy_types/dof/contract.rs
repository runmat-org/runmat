//! Original DOF signatures and errors, including their generic component labels.
use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const DOF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dof",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed structural translation or rotation degree of freedom.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimensioned displacement or rotation value matching the DOF family.",
    },
];
const NO_ARG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.component()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const DOF_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "entry = fea.StructuralDofValue.component(dof, value)",
    inputs: &DOF_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DOF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NO_ARG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const DOF_VALUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOF_VALUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
