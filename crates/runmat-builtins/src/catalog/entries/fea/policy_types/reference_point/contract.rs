//! Original reference-point descriptor; quantity validation stays in Runtime.
use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;
const REFERENCE_POINT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference-point identifier.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Position coordinates.",
    },
];
const REFERENCE_POINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "point = fea.ReferencePoint.fromCoordinates(id, position)",
    inputs: &REFERENCE_POINT_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const REFERENCE_POINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REFERENCE_POINT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
