//! Signatures, parameters, and integer facts for `fea.StructuralDofValue` constructors.
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

pub const FEA_STRUCTURAL_DOF_VALUE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const TRANSLATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "dof",
        BuiltinParamType::Any,
        "Translational DOF from fea.StructuralDof.ux/uy/uz.",
    ),
    required(
        "value",
        BuiltinParamType::Any,
        "Prescribed displacement as fea.Displacement (m).",
    ),
];
const TRANSLATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "entry",
    BuiltinParamType::Any,
    "fea.StructuralTranslationDofValue object.",
)];
const TRANSLATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "entry = fea.StructuralDofValue.translation(dof, value)",
    inputs: &TRANSLATION_INPUTS,
    outputs: &TRANSLATION_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_VALUE_TRANSLATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSLATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ROTATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "dof",
        BuiltinParamType::Any,
        "Rotational DOF from fea.StructuralDof.rx/ry/rz.",
    ),
    required(
        "value",
        BuiltinParamType::Any,
        "Prescribed rotation as fea.Rotation (rad).",
    ),
];
const ROTATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "entry",
    BuiltinParamType::Any,
    "fea.StructuralRotationDofValue object.",
)];
const ROTATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "entry = fea.StructuralDofValue.rotation(dof, value)",
    inputs: &ROTATION_INPUTS,
    outputs: &ROTATION_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_VALUE_ROTATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROTATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
