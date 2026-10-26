//! Signatures, parameters, and integer facts for `fea.ConjugateHeatTransferInterface` constructors.
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

pub const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA interface constructors accept identifiers, immutable geometry.Region values, and typed interface parameters; they have no direct integer input role.",
};

const PERFECT_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "fluidRegion",
        BuiltinParamType::Any,
        "Fluid-side face region (geometry.Region).",
    ),
    required(
        "solidRegion",
        BuiltinParamType::Any,
        "Solid-side face region (geometry.Region).",
    ),
];
const PERFECT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.ConjugateHeatTransferInterface object.",
)];
const PERFECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.ConjugateHeatTransferInterface.perfect(id, fluidRegion, solidRegion)",
    inputs: &PERFECT_INPUTS,
    outputs: &PERFECT_OUTPUTS,
}];
pub const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_PERFECT_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PERFECT_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const CONDUCTANCE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "fluidRegion",
        BuiltinParamType::Any,
        "Fluid-side face region (geometry.Region).",
    ),
    required(
        "solidRegion",
        BuiltinParamType::Any,
        "Solid-side face region (geometry.Region).",
    ),
    required(
        "conductance",
        BuiltinParamType::Any,
        "Interface conductance as positive fea.HeatTransferCoefficient (W/(m^2 K)).",
    ),
];
const CONDUCTANCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.ConjugateHeatTransferInterface object.",
)];
const CONDUCTANCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.ConjugateHeatTransferInterface.conductance(id, fluidRegion, solidRegion, conductance)",
    inputs: &CONDUCTANCE_INPUTS,
    outputs: &CONDUCTANCE_OUTPUTS,
}];
pub const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CONDUCTANCE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &CONDUCTANCE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const RESISTANCE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "fluidRegion",
        BuiltinParamType::Any,
        "Fluid-side face region (geometry.Region).",
    ),
    required(
        "solidRegion",
        BuiltinParamType::Any,
        "Solid-side face region (geometry.Region).",
    ),
    required(
        "resistance",
        BuiltinParamType::Any,
        "Area-specific resistance as positive fea.ThermalContactResistance (m^2 K/W).",
    ),
];
const RESISTANCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.ConjugateHeatTransferInterface object.",
)];
const RESISTANCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.ConjugateHeatTransferInterface.resistance(id, fluidRegion, solidRegion, resistance)",
    inputs: &RESISTANCE_INPUTS,
    outputs: &RESISTANCE_OUTPUTS,
}];
pub const FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_RESISTANCE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &RESISTANCE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
