//! Signatures, parameters, and integer facts for `fea.PeriodicTransform` constructors.
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

pub const FEA_PERIODIC_TRANSFORM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const IDENTITY_INPUTS: [BuiltinParamDescriptor; 0] = [];
const IDENTITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "transform",
    BuiltinParamType::Any,
    "fea.IdentityPeriodicTransform object.",
)];
const IDENTITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.identity()",
    inputs: &IDENTITY_INPUTS,
    outputs: &IDENTITY_OUTPUTS,
}];
pub const FEA_PERIODIC_TRANSFORM_IDENTITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IDENTITY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TRANSLATION_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "translation",
    BuiltinParamType::Any,
    "Offset from primary to secondary as fea.Translation (m).",
)];
const TRANSLATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "transform",
    BuiltinParamType::Any,
    "fea.TranslationPeriodicTransform object.",
)];
const TRANSLATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.translation(translation)",
    inputs: &TRANSLATION_INPUTS,
    outputs: &TRANSLATION_OUTPUTS,
}];
pub const FEA_PERIODIC_TRANSFORM_TRANSLATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSLATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ROTATION_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "origin",
        BuiltinParamType::Any,
        "Point on the rotation axis as fea.Position (m).",
    ),
    required(
        "axis",
        BuiltinParamType::Any,
        "Rotation axis as fea.Direction.",
    ),
    required(
        "angle",
        BuiltinParamType::Any,
        "Rotation from primary to secondary as fea.Rotation.",
    ),
];
const ROTATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "transform",
    BuiltinParamType::Any,
    "fea.RotationPeriodicTransform object.",
)];
const ROTATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.rotation(origin, axis, angle)",
    inputs: &ROTATION_INPUTS,
    outputs: &ROTATION_OUTPUTS,
}];
pub const FEA_PERIODIC_TRANSFORM_ROTATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROTATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
