//! Signatures, parameters, and integer facts for `fea.ThermalExcitation` constructors.
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

pub const FEA_THERMAL_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const HEAT_GENERATION_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.HeatGenerationSignal carrying the action's value.",
    ),
];
const HEAT_GENERATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ThermalExcitation action object.",
)];
const HEAT_GENERATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ThermalExcitation.heatGeneration(id, region, signal)",
    inputs: &HEAT_GENERATION_INPUTS,
    outputs: &HEAT_GENERATION_OUTPUTS,
}];
pub const FEA_THERMAL_EXCITATION_HEAT_GENERATION_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &HEAT_GENERATION_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const HEAT_FLUX_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.HeatFluxSignal carrying the action's value.",
    ),
];
const HEAT_FLUX_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ThermalExcitation action object.",
)];
const HEAT_FLUX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.ThermalExcitation.heatFlux(id, region, signal)",
    inputs: &HEAT_FLUX_INPUTS,
    outputs: &HEAT_FLUX_OUTPUTS,
}];
pub const FEA_THERMAL_EXCITATION_HEAT_FLUX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEAT_FLUX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TEMPERATURE_FIELD_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.TemperatureSignal carrying the action's value.",
    ),
];
const TEMPERATURE_FIELD_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ThermalExcitation action object.",
)];
const TEMPERATURE_FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "action = fea.ThermalExcitation.temperatureField(id, region, signal)",
        inputs: &TEMPERATURE_FIELD_INPUTS,
        outputs: &TEMPERATURE_FIELD_OUTPUTS,
    }];
pub const FEA_THERMAL_EXCITATION_TEMPERATURE_FIELD_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &TEMPERATURE_FIELD_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const AMBIENT_TEMPERATURE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.TemperatureSignal carrying the action's value.",
    ),
];
const AMBIENT_TEMPERATURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.ThermalExcitation action object.",
)];
const AMBIENT_TEMPERATURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "action = fea.ThermalExcitation.ambientTemperature(id, region, signal)",
        inputs: &AMBIENT_TEMPERATURE_INPUTS,
        outputs: &AMBIENT_TEMPERATURE_OUTPUTS,
    }];
pub const FEA_THERMAL_EXCITATION_AMBIENT_TEMPERATURE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &AMBIENT_TEMPERATURE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
