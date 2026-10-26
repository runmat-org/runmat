//! Signatures, parameters, and integer facts for `fea.ThermalConstraint` constructors.
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

pub const FEA_THERMAL_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

pub const FEA_THERMAL_CONSTRAINT_PRESCRIBED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The prescribed-temperature constructor accepts an identifier, geometry.Region, and already-dimensioned fea.Temperature value; it has no direct integer input role.",
};

const PRESCRIBED_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "temperature",
        BuiltinParamType::Any,
        "Absolute temperature as fea.Temperature.",
    ),
];
const PRESCRIBED_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ThermalConstraint object.",
)];
const PRESCRIBED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.ThermalConstraint.prescribed(id, region, temperature)",
    inputs: &PRESCRIBED_INPUTS,
    outputs: &PRESCRIBED_OUTPUTS,
}];
pub const FEA_THERMAL_CONSTRAINT_PRESCRIBED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESCRIBED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TIE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region, distinct from primary.",
    ),
    required(
        "pairing",
        BuiltinParamType::Any,
        "fea.PeriodicPairing policy that matches primary and secondary entities.",
    ),
];
const TIE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ThermalConstraint object.",
)];
const TIE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.ThermalConstraint.tie(id, primary, secondary, pairing)",
    inputs: &TIE_INPUTS,
    outputs: &TIE_OUTPUTS,
}];
pub const FEA_THERMAL_CONSTRAINT_TIE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PERIODIC_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region, distinct from primary.",
    ),
    required(
        "transform",
        BuiltinParamType::Any,
        "fea.PeriodicTransform mapping primary geometry onto secondary geometry.",
    ),
    required(
        "pairing",
        BuiltinParamType::Any,
        "fea.PeriodicPairing policy that matches primary and secondary entities.",
    ),
];
const PERIODIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ThermalConstraint object.",
)];
const PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label:
        "constraint = fea.ThermalConstraint.periodic(id, primary, secondary, transform, pairing)",
    inputs: &PERIODIC_INPUTS,
    outputs: &PERIODIC_OUTPUTS,
}];
pub const FEA_THERMAL_CONSTRAINT_PERIODIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const AFFINE_JUMP_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region, distinct from primary.",
    ),
    required(
        "transform",
        BuiltinParamType::Any,
        "fea.PeriodicTransform mapping primary geometry onto secondary geometry.",
    ),
    required(
        "pairing",
        BuiltinParamType::Any,
        "fea.PeriodicPairing policy that matches primary and secondary entities.",
    ),
    required(
        "jump",
        BuiltinParamType::Any,
        "Temperature jump as fea.TemperatureDifference (K).",
    ),
];
const AFFINE_JUMP_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ThermalConstraint object.",
)];
const AFFINE_JUMP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.ThermalConstraint.affineJump(id, primary, secondary, transform, pairing, jump)",
    inputs: &AFFINE_JUMP_INPUTS,
    outputs: &AFFINE_JUMP_OUTPUTS,
}];
pub const FEA_THERMAL_CONSTRAINT_AFFINE_JUMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AFFINE_JUMP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
