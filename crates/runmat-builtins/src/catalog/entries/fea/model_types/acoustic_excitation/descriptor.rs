//! Signatures, parameters, and integer facts for `fea.AcousticExcitation` constructors.
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

pub const FEA_ACOUSTIC_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const PRESSURE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.PressureSignal carrying the action's value.",
    ),
];
const PRESSURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.AcousticExcitation action object.",
)];
const PRESSURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.AcousticExcitation.pressure(id, region, signal)",
    inputs: &PRESSURE_INPUTS,
    outputs: &PRESSURE_OUTPUTS,
}];
pub const FEA_ACOUSTIC_EXCITATION_PRESSURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESSURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const NORMAL_VELOCITY_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.VelocitySignal carrying the action's value.",
    ),
];
const NORMAL_VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.AcousticExcitation action object.",
)];
const NORMAL_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.AcousticExcitation.normalVelocity(id, region, signal)",
    inputs: &NORMAL_VELOCITY_INPUTS,
    outputs: &NORMAL_VELOCITY_OUTPUTS,
}];
pub const FEA_ACOUSTIC_EXCITATION_NORMAL_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &NORMAL_VELOCITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const VOLUME_VELOCITY_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.VolumeFlowSignal carrying the action's value.",
    ),
];
const VOLUME_VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.AcousticExcitation action object.",
)];
const VOLUME_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.AcousticExcitation.volumeVelocity(id, region, signal)",
    inputs: &VOLUME_VELOCITY_INPUTS,
    outputs: &VOLUME_VELOCITY_OUTPUTS,
}];
pub const FEA_ACOUSTIC_EXCITATION_VOLUME_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &VOLUME_VELOCITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
