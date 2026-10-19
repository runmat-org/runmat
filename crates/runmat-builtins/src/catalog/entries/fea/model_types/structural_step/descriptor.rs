//! Signatures, parameters, and integer facts for `fea.StructuralStep` constructors.
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

pub const FEA_STRUCTURAL_STEP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA step constructors accept only a text identifier. Family, procedure, solution space, and default independent domain are fixed by the concrete class method.",
};

const STATIC_STEP_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const STATIC_STEP_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.StructuralStep object.",
)];
const STATIC_STEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.StructuralStep.static(id)",
    inputs: &STATIC_STEP_INPUTS,
    outputs: &STATIC_STEP_OUTPUTS,
}];
pub const FEA_STRUCTURAL_STEP_STATIC_STEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STATIC_STEP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MODAL_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const MODAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.StructuralStep object.",
)];
const MODAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.StructuralStep.modal(id)",
    inputs: &MODAL_INPUTS,
    outputs: &MODAL_OUTPUTS,
}];
pub const FEA_STRUCTURAL_STEP_MODAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const HARMONIC_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const HARMONIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.StructuralStep object.",
)];
const HARMONIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.StructuralStep.harmonic(id)",
    inputs: &HARMONIC_INPUTS,
    outputs: &HARMONIC_OUTPUTS,
}];
pub const FEA_STRUCTURAL_STEP_HARMONIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HARMONIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TRANSIENT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const TRANSIENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.StructuralStep object.",
)];
const TRANSIENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.StructuralStep.transient(id)",
    inputs: &TRANSIENT_INPUTS,
    outputs: &TRANSIENT_OUTPUTS,
}];
pub const FEA_STRUCTURAL_STEP_TRANSIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const NONLINEAR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Procedure-step identifier (nonempty text).",
)];
const NONLINEAR_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "fea.StructuralStep object.",
)];
const NONLINEAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.StructuralStep.nonlinear(id)",
    inputs: &NONLINEAR_INPUTS,
    outputs: &NONLINEAR_OUTPUTS,
}];
pub const FEA_STRUCTURAL_STEP_NONLINEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NONLINEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
