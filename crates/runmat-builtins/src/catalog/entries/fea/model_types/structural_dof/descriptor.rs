//! Signatures, parameters, and integer facts for `fea.StructuralDof` constructors.
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

pub const FEA_STRUCTURAL_DOF_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const UX_INPUTS: [BuiltinParamDescriptor; 0] = [];
const UX_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralTranslationDof object.",
)];
const UX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.ux()",
    inputs: &UX_INPUTS,
    outputs: &UX_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_UX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const UY_INPUTS: [BuiltinParamDescriptor; 0] = [];
const UY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralTranslationDof object.",
)];
const UY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.uy()",
    inputs: &UY_INPUTS,
    outputs: &UY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_UY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const UZ_INPUTS: [BuiltinParamDescriptor; 0] = [];
const UZ_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralTranslationDof object.",
)];
const UZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.uz()",
    inputs: &UZ_INPUTS,
    outputs: &UZ_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_UZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const RX_INPUTS: [BuiltinParamDescriptor; 0] = [];
const RX_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralRotationDof object.",
)];
const RX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.rx()",
    inputs: &RX_INPUTS,
    outputs: &RX_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_RX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const RY_INPUTS: [BuiltinParamDescriptor; 0] = [];
const RY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralRotationDof object.",
)];
const RY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.ry()",
    inputs: &RY_INPUTS,
    outputs: &RY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_RY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const RZ_INPUTS: [BuiltinParamDescriptor; 0] = [];
const RZ_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "dof",
    BuiltinParamType::Any,
    "fea.StructuralRotationDof object.",
)];
const RZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.rz()",
    inputs: &RZ_INPUTS,
    outputs: &RZ_OUTPUTS,
}];
pub const FEA_STRUCTURAL_DOF_RZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
