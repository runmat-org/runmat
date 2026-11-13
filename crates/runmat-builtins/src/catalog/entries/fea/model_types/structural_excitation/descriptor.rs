//! Signatures, parameters, and integer facts for `fea.StructuralExcitation` constructors.
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

pub const FEA_STRUCTURAL_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const FORCE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.ForceSignal carrying the action's value.",
    ),
];
const FORCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.StructuralExcitation action object.",
)];
const FORCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.StructuralExcitation.force(id, region, signal)",
    inputs: &FORCE_INPUTS,
    outputs: &FORCE_OUTPUTS,
}];
pub const FEA_STRUCTURAL_EXCITATION_FORCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FORCE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MOMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.MomentSignal carrying the action's value.",
    ),
];
const MOMENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.StructuralExcitation action object.",
)];
const MOMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.StructuralExcitation.moment(id, region, signal)",
    inputs: &MOMENT_INPUTS,
    outputs: &MOMENT_OUTPUTS,
}];
pub const FEA_STRUCTURAL_EXCITATION_MOMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MOMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TRACTION_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.TractionSignal carrying the action's value.",
    ),
];
const TRACTION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.StructuralExcitation action object.",
)];
const TRACTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.StructuralExcitation.traction(id, region, signal)",
    inputs: &TRACTION_INPUTS,
    outputs: &TRACTION_OUTPUTS,
}];
pub const FEA_STRUCTURAL_EXCITATION_TRACTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRACTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
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
    "fea.StructuralExcitation action object.",
)];
const PRESSURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.StructuralExcitation.pressure(id, region, signal)",
    inputs: &PRESSURE_INPUTS,
    outputs: &PRESSURE_OUTPUTS,
}];
pub const FEA_STRUCTURAL_EXCITATION_PRESSURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESSURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const BODY_FORCE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.BodyForceSignal carrying the action's value.",
    ),
];
const BODY_FORCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.StructuralExcitation action object.",
)];
const BODY_FORCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.StructuralExcitation.bodyForce(id, region, signal)",
    inputs: &BODY_FORCE_INPUTS,
    outputs: &BODY_FORCE_OUTPUTS,
}];
pub const FEA_STRUCTURAL_EXCITATION_BODY_FORCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BODY_FORCE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
