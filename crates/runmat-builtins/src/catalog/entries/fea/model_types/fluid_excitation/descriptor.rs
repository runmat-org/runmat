//! Signatures, parameters, and integer facts for `fea.FluidExcitation` constructors.
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

pub const FEA_FLUID_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const INLET_VELOCITY_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const INLET_VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.FluidExcitation action object.",
)];
const INLET_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.FluidExcitation.inletVelocity(id, region, signal)",
    inputs: &INLET_VELOCITY_INPUTS,
    outputs: &INLET_VELOCITY_OUTPUTS,
}];
pub const FEA_FLUID_EXCITATION_INLET_VELOCITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INLET_VELOCITY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MASS_FLOW_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.MassFlowSignal carrying the action's value.",
    ),
];
const MASS_FLOW_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.FluidExcitation action object.",
)];
const MASS_FLOW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.FluidExcitation.massFlow(id, region, signal)",
    inputs: &MASS_FLOW_INPUTS,
    outputs: &MASS_FLOW_OUTPUTS,
}];
pub const FEA_FLUID_EXCITATION_MASS_FLOW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MASS_FLOW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const WALL_MOTION_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const WALL_MOTION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.FluidExcitation action object.",
)];
const WALL_MOTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.FluidExcitation.wallMotion(id, region, signal)",
    inputs: &WALL_MOTION_INPUTS,
    outputs: &WALL_MOTION_OUTPUTS,
}];
pub const FEA_FLUID_EXCITATION_WALL_MOTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WALL_MOTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
