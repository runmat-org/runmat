//! Signatures, parameters, and integer facts for `fea.AcousticStep` constructors.
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

pub const FEA_ACOUSTIC_STEP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA step constructors accept only a text identifier. Family, procedure, solution space, and default independent domain are fixed by the concrete class method.",
};

const HARMONIC_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const HARMONIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.AcousticStep object.",
)];
const HARMONIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.AcousticStep.harmonic(id)",
    inputs: &HARMONIC_INPUTS,
    outputs: &HARMONIC_OUTPUTS,
}];
pub const FEA_ACOUSTIC_STEP_HARMONIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HARMONIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
