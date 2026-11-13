//! Signatures, parameters, and integer facts for `fea.ElectromagneticDomain` constructors.
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

pub const FEA_ELECTROMAGNETIC_DOMAIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA domain constructors accept dimensioned quantity objects; integer handling belongs to those quantity constructors.",
};

const AT_FREQUENCY_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frequency",
    BuiltinParamType::Any,
    "Positive excitation frequency as fea.Frequency (Hz).",
)];
const AT_FREQUENCY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "domain",
    BuiltinParamType::Any,
    "fea.ElectromagneticDomain object.",
)];
const AT_FREQUENCY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "domain = fea.ElectromagneticDomain.atFrequency(frequency)",
    inputs: &AT_FREQUENCY_INPUTS,
    outputs: &AT_FREQUENCY_OUTPUTS,
}];
pub const FEA_ELECTROMAGNETIC_DOMAIN_AT_FREQUENCY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &AT_FREQUENCY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
