//! Signatures, parameters, and integer facts for `fea.StepObjectRef` constructors.
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

pub const FEA_STEP_OBJECT_REF_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed activation constructors accept identifiers and typed FEA objects only.",
    };

const CONSTRAINT_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of a constraint in the model (nonempty text).",
)];
const CONSTRAINT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const CONSTRAINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = fea.StepObjectRef.constraint(id)",
    inputs: &CONSTRAINT_INPUTS,
    outputs: &CONSTRAINT_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_REF_CONSTRAINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONSTRAINT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const BOUNDARY_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of a boundary in the model (nonempty text).",
)];
const BOUNDARY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const BOUNDARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = fea.StepObjectRef.boundary(id)",
    inputs: &BOUNDARY_INPUTS,
    outputs: &BOUNDARY_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_REF_BOUNDARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOUNDARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CONNECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of a connector in the model (nonempty text).",
)];
const CONNECTOR_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const CONNECTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = fea.StepObjectRef.connector(id)",
    inputs: &CONNECTOR_INPUTS,
    outputs: &CONNECTOR_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_REF_CONNECTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONNECTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTERFACE_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of an interface in the model (nonempty text).",
)];
const INTERFACE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const INTERFACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = fea.StepObjectRef.interface(id)",
    inputs: &INTERFACE_INPUTS,
    outputs: &INTERFACE_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_REF_INTERFACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERFACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const EXCITATION_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of an excitation action in the model (nonempty text).",
)];
const EXCITATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const EXCITATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = fea.StepObjectRef.excitation(id)",
    inputs: &EXCITATION_INPUTS,
    outputs: &EXCITATION_OUTPUTS,
}];
pub const FEA_STEP_OBJECT_REF_EXCITATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXCITATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INITIAL_CONDITION_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of an initial condition in the model (nonempty text).",
)];
const INITIAL_CONDITION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const INITIAL_CONDITION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "ref = fea.StepObjectRef.initialCondition(id)",
        inputs: &INITIAL_CONDITION_INPUTS,
        outputs: &INITIAL_CONDITION_OUTPUTS,
    }];
pub const FEA_STEP_OBJECT_REF_INITIAL_CONDITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INITIAL_CONDITION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MATERIAL_PARAMETER_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Id of a material parameter in the model (nonempty text).",
)];
const MATERIAL_PARAMETER_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "ref",
    BuiltinParamType::Any,
    "fea.StepObjectRef object.",
)];
const MATERIAL_PARAMETER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "ref = fea.StepObjectRef.materialParameter(id)",
        inputs: &MATERIAL_PARAMETER_INPUTS,
        outputs: &MATERIAL_PARAMETER_OUTPUTS,
    }];
pub const FEA_STEP_OBJECT_REF_MATERIAL_PARAMETER_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &MATERIAL_PARAMETER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
