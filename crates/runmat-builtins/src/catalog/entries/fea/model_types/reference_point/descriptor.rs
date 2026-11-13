//! Signatures, parameters, and integer facts for `fea.ReferencePoint` constructors.
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

pub const FEA_REFERENCE_POINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const FROM_COORDINATES_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Reference-point identifier (nonempty text).",
    ),
    required(
        "position",
        BuiltinParamType::Any,
        "Point location as fea.Position (m).",
    ),
];
const FROM_COORDINATES_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "point",
    BuiltinParamType::Any,
    "fea.ReferencePoint object.",
)];
const FROM_COORDINATES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "point = fea.ReferencePoint.fromCoordinates(id, position)",
    inputs: &FROM_COORDINATES_INPUTS,
    outputs: &FROM_COORDINATES_OUTPUTS,
}];
pub const FEA_REFERENCE_POINT_FROM_COORDINATES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FROM_COORDINATES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
