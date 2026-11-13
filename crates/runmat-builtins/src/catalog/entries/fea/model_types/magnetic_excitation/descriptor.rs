//! Signatures, parameters, and integer facts for `fea.MagneticExcitation` constructors.
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

pub const FEA_MAGNETIC_EXCITATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const VECTOR_POTENTIAL_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "fea.MagneticVectorPotentialSignal carrying the action's value.",
    ),
];
const VECTOR_POTENTIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "action",
    BuiltinParamType::Any,
    "fea.MagneticExcitation action object.",
)];
const VECTOR_POTENTIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "action = fea.MagneticExcitation.vectorPotential(id, region, signal)",
    inputs: &VECTOR_POTENTIAL_INPUTS,
    outputs: &VECTOR_POTENTIAL_OUTPUTS,
}];
pub const FEA_MAGNETIC_EXCITATION_VECTOR_POTENTIAL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &VECTOR_POTENTIAL_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
