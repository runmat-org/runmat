//! Signatures, parameters, and integer facts for `fea.StructuralMotion` constructors.
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

pub const FEA_STRUCTURAL_MOTION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const ACCELERATION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "dof",
        BuiltinParamType::Any,
        "Translational DOF from fea.StructuralDof.ux/uy/uz.",
    ),
    required(
        "acceleration",
        BuiltinParamType::Any,
        "Global-frame, real, three-component time-domain fea.AccelerationSignal.",
    ),
    required(
        "initialDisplacement",
        BuiltinParamType::Any,
        "Support displacement at the start of the step as fea.Displacement (m).",
    ),
    required(
        "initialVelocity",
        BuiltinParamType::Any,
        "Support velocity at the start of the step as fea.Velocity (m/s).",
    ),
];
const ACCELERATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "motion",
    BuiltinParamType::Any,
    "fea.StructuralTranslationMotion object.",
)];
const ACCELERATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "motion = fea.StructuralMotion.acceleration(dof, acceleration, initialDisplacement, initialVelocity)",
    inputs: &ACCELERATION_INPUTS,
    outputs: &ACCELERATION_OUTPUTS,
}];
pub const FEA_STRUCTURAL_MOTION_ACCELERATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACCELERATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
