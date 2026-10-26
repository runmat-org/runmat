//! Signatures, parameters, and integer facts for `fea.MechanicalDamping` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const UNDAMPED_INPUTS: [BuiltinParamDescriptor; 0] = [];
const UNDAMPED_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "damping",
    BuiltinParamType::Any,
    "fea.MechanicalDamping object.",
)];
const UNDAMPED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "damping = fea.MechanicalDamping.undamped()",
    inputs: &UNDAMPED_INPUTS,
    outputs: &UNDAMPED_OUTPUTS,
}];
pub const FEA_MECHANICAL_DAMPING_UNDAMPED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNDAMPED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const RAYLEIGH_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "massCoefficient",
        BuiltinParamType::Any,
        "Mass-proportional coefficient alpha as fea.RayleighMassCoefficient (1/s).",
    ),
    required(
        "stiffnessCoefficient",
        BuiltinParamType::Any,
        "Stiffness-proportional coefficient beta as fea.RayleighStiffnessCoefficient (s).",
    ),
];
const RAYLEIGH_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "damping",
    BuiltinParamType::Any,
    "fea.MechanicalDamping object.",
)];
const RAYLEIGH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "damping = fea.MechanicalDamping.rayleigh(massCoefficient, stiffnessCoefficient)",
    inputs: &RAYLEIGH_INPUTS,
    outputs: &RAYLEIGH_OUTPUTS,
}];
pub const FEA_MECHANICAL_DAMPING_RAYLEIGH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RAYLEIGH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
