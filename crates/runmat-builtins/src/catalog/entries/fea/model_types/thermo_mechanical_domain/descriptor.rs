//! Signatures, parameters, and integer facts for `fea.ThermoMechanicalDomain` constructors.
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

pub const FEA_THERMO_MECHANICAL_DOMAIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA domain constructors accept dimensioned quantity objects; integer handling belongs to those quantity constructors.",
};

const AT_REFERENCE_TEMPERATURE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "referenceTemperature",
    BuiltinParamType::Any,
    "Reference (stress-free or property-reference) temperature as fea.Temperature.",
)];
const AT_REFERENCE_TEMPERATURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "domain",
    BuiltinParamType::Any,
    "fea.ThermoMechanicalDomain object.",
)];
const AT_REFERENCE_TEMPERATURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "domain = fea.ThermoMechanicalDomain.atReferenceTemperature(referenceTemperature)",
        inputs: &AT_REFERENCE_TEMPERATURE_INPUTS,
        outputs: &AT_REFERENCE_TEMPERATURE_OUTPUTS,
    }];
pub const FEA_THERMO_MECHANICAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &AT_REFERENCE_TEMPERATURE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
