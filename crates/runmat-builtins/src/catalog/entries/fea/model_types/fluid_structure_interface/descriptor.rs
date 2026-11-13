//! Signatures, parameters, and integer facts for `fea.FluidStructureInterface` constructors.
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

pub const FEA_FLUID_STRUCTURE_INTERFACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA interface constructors accept identifiers, immutable geometry.Region values, and typed interface parameters; they have no direct integer input role.",
};

const NO_SLIP_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier (nonempty text).",
    ),
    required(
        "fluidRegion",
        BuiltinParamType::Any,
        "Fluid-side face region (geometry.Region).",
    ),
    required(
        "structureRegion",
        BuiltinParamType::Any,
        "Structure-side face region (geometry.Region).",
    ),
];
const NO_SLIP_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interface",
    BuiltinParamType::Any,
    "fea.FluidStructureInterface object.",
)];
const NO_SLIP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.FluidStructureInterface.noSlip(id, fluidRegion, structureRegion)",
    inputs: &NO_SLIP_INPUTS,
    outputs: &NO_SLIP_OUTPUTS,
}];
pub const FEA_FLUID_STRUCTURE_INTERFACE_NO_SLIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NO_SLIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
