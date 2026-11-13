//! Signatures, parameters, and integer facts for `fea.ElectricalExcitation` constructors.
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

pub const FEA_ELECTRICAL_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const TERMINAL_VOLTAGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "fea.ElectricPotentialSignal carrying the action's value.",
    ),
];
const TERMINAL_VOLTAGE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ElectricalExcitation action object.",
)];
const TERMINAL_VOLTAGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ElectricalExcitation.terminalVoltage(id, region, signal)",
    inputs: &TERMINAL_VOLTAGE_INPUTS,
    outputs: &TERMINAL_VOLTAGE_OUTPUTS,
}];
pub const FEA_ELECTRICAL_EXCITATION_TERMINAL_VOLTAGE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &TERMINAL_VOLTAGE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const CURRENT_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "fea.ElectricCurrentSignal carrying the action's value.",
    ),
];
const CURRENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ElectricalExcitation action object.",
)];
const CURRENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ElectricalExcitation.current(id, region, signal)",
    inputs: &CURRENT_INPUTS,
    outputs: &CURRENT_OUTPUTS,
}];
pub const FEA_ELECTRICAL_EXCITATION_CURRENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CURRENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CHARGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "fea.ElectricChargeSignal carrying the action's value.",
    ),
];
const CHARGE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ElectricalExcitation action object.",
)];
const CHARGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ElectricalExcitation.charge(id, region, signal)",
    inputs: &CHARGE_INPUTS,
    outputs: &CHARGE_OUTPUTS,
}];
pub const FEA_ELECTRICAL_EXCITATION_CHARGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHARGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CURRENT_DENSITY_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "fea.CurrentDensitySignal carrying the action's value.",
    ),
];
const CURRENT_DENSITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ElectricalExcitation action object.",
)];
const CURRENT_DENSITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ElectricalExcitation.currentDensity(id, region, signal)",
    inputs: &CURRENT_DENSITY_INPUTS,
    outputs: &CURRENT_DENSITY_OUTPUTS,
}];
pub const FEA_ELECTRICAL_EXCITATION_CURRENT_DENSITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &CURRENT_DENSITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
