//! Signatures, parameters, and integer facts for `fea.ContactInterface` constructors.
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

pub const FEA_CONTACT_INTERFACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA interface constructors accept identifiers, immutable geometry.Region values, and typed interface parameters; they have no direct integer input role.",
};

const FRICTIONLESS_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "primary",
        BuiltinParamType::Any,
        "Primary contact surface (geometry.Region).",
    ),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary contact surface (geometry.Region).",
    ),
];
const FRICTIONLESS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.ContactInterface object.",
)];
const FRICTIONLESS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.ContactInterface.frictionless(id, primary, secondary)",
    inputs: &FRICTIONLESS_INPUTS,
    outputs: &FRICTIONLESS_OUTPUTS,
}];
pub const FEA_CONTACT_INTERFACE_FRICTIONLESS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FRICTIONLESS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const COULOMB_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "primary",
        BuiltinParamType::Any,
        "Primary contact surface (geometry.Region).",
    ),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary contact surface (geometry.Region).",
    ),
    required(
        "friction",
        BuiltinParamType::Any,
        "Coulomb coefficient as a positive fea.FrictionCoefficient.",
    ),
];
const COULOMB_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.ContactInterface object.",
)];
const COULOMB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.ContactInterface.coulomb(id, primary, secondary, friction)",
    inputs: &COULOMB_INPUTS,
    outputs: &COULOMB_OUTPUTS,
}];
pub const FEA_CONTACT_INTERFACE_COULOMB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COULOMB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
